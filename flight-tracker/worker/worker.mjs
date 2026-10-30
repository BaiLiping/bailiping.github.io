import {lookup,LookupError} from './api.mjs';
import live from '../live-core.js';

const json = (value,status=200,headers={}) => new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store',...headers}});
const failure = error => json({error:error instanceof LookupError?error.message:'The flight lookup could not be completed. Please try again.',code:error.code || 'lookup_failed'},error.status || 502);

export default {
  async fetch(request,env) {
    const url=new URL(request.url);
    const origin=request.headers.get('Origin');
    const allowed=(env.ALLOWED_ORIGINS || 'https://bailiping.com').split(',').map(s=>s.trim());
    if(!origin || !allowed.includes(origin)) return json({error:'This endpoint is available from the flight tracker page.'},403);
    const cors={'Access-Control-Allow-Origin':origin,'Vary':'Origin','Access-Control-Allow-Methods':'POST, GET, OPTIONS','Access-Control-Allow-Headers':'Content-Type','Access-Control-Max-Age':'600'};
    let response;
    try {
      if(request.method==='OPTIONS') return new Response(null,{status:204,headers:cors});
      if(url.pathname==='/health' && request.method==='GET') return json({ready:!!env.OPENAI_API_KEY && !!env.LIVE_CHECKS,source:'Flightradar24'},200,cors);
      if(url.pathname!=='/check' || request.method!=='POST') return json({error:'Not found.'},404,cors);
      if(!env.OPENAI_API_KEY || !env.LIVE_CHECKS) throw new LookupError('Live checks are not connected yet.',503,'not_configured');
      if(!(request.headers.get('Content-Type') || '').toLowerCase().startsWith('application/json')) return json({error:'Send a JSON flight selection.'},415,cors);
      if(Number(request.headers.get('Content-Length'))>2048) return json({error:'The flight selection is too large.'},413,cors);
      const text=await request.text();
      if(text.length>2048) return json({error:'The flight selection is too large.'},413,cors);
      let query;
      try { query=live.selection(JSON.parse(text)); } catch { return json({error:'Enter a valid flight, departure airport, and date.'},400,cors); }
      // The only stateful entry point is this one named object, so quotas are global.
      const id=env.LIVE_CHECKS.idFromName('flight-checks-v1');
      const headers={'Content-Type':'application/json'};
      headers['X-Visitor']=request.headers.get('CF-Connecting-IP') || 'unknown';
      response=await env.LIVE_CHECKS.get(id).fetch(new Request('https://internal/check',{method:'POST',headers,body:JSON.stringify(query)}));
    } catch(error) { response=failure(error); }
    const headers=new Headers(response.headers);
    for(const [name,value] of Object.entries(cors)) headers.set(name,value);
    return new Response(response.body,{status:response.status,headers});
  }
};

export class LiveChecks {
  constructor(ctx,env) {
    this.ctx=ctx;this.env=env;this.pending=new Map();
    this.ready=ctx.blockConcurrencyWhile(async()=>{
      if(await ctx.storage.get('input-only-v1')) return;
      // Remove the retired result cache and archive, preserving usage counters.
      for(const prefix of ['cache:','archive:']) {
        let entries;
        do {
          entries=await ctx.storage.list({prefix,limit:128});
          if(entries.size) await ctx.storage.delete([...entries.keys()]);
        } while(entries.size===128);
      }
      await ctx.storage.put('input-only-v1',true);
    });
  }
  async lookup(query) { return lookup(query,this.env); }
  async fetch(request) {
    try {
      await this.ready;
      const query=live.selection(await request.json());
      const key=[query.flight,query.airport,query.date].join(':');
      if(!this.pending.has(key)) {
        const promise=this.check(query,request.headers.get('X-Visitor') || 'unknown').finally(()=>this.pending.delete(key));
        this.pending.set(key,promise);
      }
      return json(await this.pending.get(key));
    } catch(error) { return failure(error); }
  }
  async reserve(visitor,now) {
    const day=new Date(now).toISOString().slice(0,10);
    const hour=Math.floor(now/3600000);
    const daily=Math.min(50,Math.max(1,Number(this.env.MAX_DAILY_CHECKS) || 10));
    const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(day+':'+visitor));
    const visitorId=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
    await this.ctx.storage.transaction(async txn=>{
      const previous=await txn.get('budget');
      const budget=previous?.day===day?previous:{day,total:0,visitors:{}};
      const count=budget.visitors[visitorId]?.hour===hour?budget.visitors[visitorId].count:0;
      if(budget.total>=daily) throw new LookupError('Today’s flight-lookup allowance has been reached. Please try again tomorrow.',429,'daily_limit');
      if(count>=3) throw new LookupError('Please wait before requesting another flight lookup.',429,'rate_limit');
      budget.total++;budget.visitors[visitorId]={hour,count:count+1};
      await txn.put('budget',budget);
    });
  }
  async check(query,visitor) {
    await this.ready;
    await this.reserve(visitor,Date.now());
    return this.lookup(query);
  }
}
