/** Deterministic numerical examples. No random draws or external dependencies. */
const check=(ok,msg)=>{if(!ok)throw new RangeError(msg)};
export function normal(x,mean=0,sd=1){check(sd>0&&Number.isFinite(sd),'sd must be positive and finite');return Math.exp(-.5*((x-mean)/sd)**2)/(sd*Math.sqrt(2*Math.PI));}
export function choose(n,k){check(Number.isInteger(n)&&n>=0&&Number.isInteger(k)&&k>=0&&k<=n,'invalid binomial counts');let c=1;for(let i=1;i<=Math.min(k,n-k);i++)c=c*(n-i+1)/i;return c;}
export function binomial(n,k,t){check(t>=0&&t<=1,'theta must be in [0,1]');return choose(n,k)*t**k*(1-t)**(n-k);}
export function betaPdf(t,a,b){check(Number.isInteger(a)&&Number.isInteger(b)&&a>=1&&b>=1,'integer positive beta shapes required');check(t>=0&&t<=1,'theta must be in [0,1]');return (a+b-1)*choose(a+b-2,a-1)*t**(a-1)*(1-t)**(b-1);}
export function coin(n,k,a=1,b=1){choose(n,k);check(n>0,'positive sample size required');betaPdf(.5,a,b);return {mle:k/n,alpha:a+k,beta:b+n-k,mean:(a+k)/(a+b+n),area:1/(n+1)};}
export function integrate(f,a,b,n=4000){check(n>0&&Number.isInteger(n)&&b>a,'invalid quadrature grid');let s=(f(a)+f(b))/2;for(let i=1;i<n;i++)s+=f(a+(b-a)*i/n);return s*(b-a)/n;}
export function gaussianUpdate(m0,p0,y,r){check(p0>0&&r>0,'variances must be positive');const p=1/(1/p0+1/r);return {mean:p*(m0/p0+y/r),variance:p};}
