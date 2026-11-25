import {writeFileSync,mkdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const dir=fileURLToPath(new URL('./figures/',import.meta.url));
mkdirSync(dir,{recursive:true});
const wrap=(w,h,title,body)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><title>${title}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9" fill="none" stroke="#60747A" stroke-width="1.7"/></marker></defs><style>text{font-family:Arial,Helvetica,sans-serif;fill:#182D33} .label{font-size:15px;font-weight:700;letter-spacing:.7px;fill:#16736E}.small{font-size:16px;fill:#60747A}.body{font-size:20px}.title{font-size:23px;font-weight:700}.mono{font-family:monospace;font-size:19px}.line{stroke:#60747A;stroke-width:1.6;fill:none}.dash{stroke-dasharray:7 6}</style><rect width="${w}" height="${h}" rx="10" fill="#FFFDFA" stroke="#D6DEDC"/>${body}</svg>`;
const save=(name,w,h,title,body)=>writeFileSync(dir+name,wrap(w,h,title,body));
save('projection.svg',562,448,'A 3D Gaussian becomes a 2D elliptical footprint',`
 <text x="24" y="35" class="label">WORLD GEOMETRY → IMAGE PLANE</text>
 <path d="M60 317L179 158L424 151L470 282Z" fill="#E4F0ED" opacity=".4"/>
 <path d="M59 317L410 119M59 317L487 320" class="line dash" opacity=".6"/>
 <path d="M342 110L469 144L493 323L366 289Z" fill="#E4F0ED" stroke="#16736E" stroke-width="1.8"/>
 <ellipse cx="211" cy="227" rx="77" ry="36" transform="rotate(-29 211 227)" fill="#16736E" fill-opacity=".14" stroke="#16736E" stroke-width="2.5"/>
 <ellipse cx="211" cy="227" rx="77" ry="15" transform="rotate(-29 211 227)" fill="none" stroke="#16736E" opacity=".6"/>
 <ellipse cx="211" cy="227" rx="23" ry="36" transform="rotate(-29 211 227)" fill="none" stroke="#16736E" opacity=".6"/>
 <ellipse cx="412" cy="218" rx="40" ry="59" transform="rotate(17 412 218)" fill="#B44F37" fill-opacity=".14" stroke="#B44F37" stroke-width="2.5"/>
 <path d="M60 317L412 218" class="line dash"/>
 <circle cx="211" cy="227" r="5" fill="#16736E"/><circle cx="412" cy="218" r="5" fill="#B44F37"/>
 <path d="M29 302H57V333H29ZM57 307L75 299V336L57 328" fill="#182D33"/>
 <text x="24" y="366" class="body">Camera</text>
 <text x="165" y="310" class="body">3D covariance Σ</text>
 <text x="358" y="353" class="body">2D covariance S</text>
 <text x="217" y="213" class="mono">μ</text><text x="424" y="213" class="mono">m</text>
 <text x="24" y="407" class="small">Schematic. The Jacobian carries local shape to the screen.</text>`);

save('composite.svg',1136,244,'The alpha weights sum to one including the background',`
 <text x="24" y="35" class="label">TWO SPLATS AT ONE PIXEL</text>
 <text x="24" y="78" class="body">Front α₁ = 0.6</text><text x="440" y="78" class="body">Behind α₂ = 0.5</text><text x="867" y="78" class="body">Background</text>
 <rect x="24" y="98" width="652.8" height="63" fill="#16736E"/>
 <rect x="676.8" y="98" width="217.6" height="63" fill="#B44F37"/>
 <rect x="894.4" y="98" width="217.6" height="63" fill="#DCE4E4"/>
 <text x="350" y="139" text-anchor="middle" style="fill:white;font-size:28px">0.60</text>
 <text x="785" y="139" text-anchor="middle" style="fill:white;font-size:28px">0.20</text>
 <text x="1003" y="139" text-anchor="middle" style="font-size:28px">0.20</text>
 <text x="24" y="203" class="body">1 × 0.6</text><text x="440" y="203" class="body">(1 − 0.6) × 0.5</text><text x="844" y="203" class="body">(1 − 0.6) × (1 − 0.5)</text>`);

save('multiview.svg',680,448,'Two camera constraints intersect in an x-z slice',`
 <text x="24" y="34" class="label">AN x–z SLICE OF THE ORTHOGRAPHIC LAB</text>
 <path d="M118 359H602M118 359V82" class="line" marker-end="url(#arrow)"/>
 <text x="608" y="367" class="body">x</text><text x="105" y="74" class="body">z</text>
 <path d="M405 87V359" stroke="#16736E" stroke-width="2" stroke-dasharray="7 5"/>
 <path d="M118 165H586" stroke="#B44F37" stroke-width="2" stroke-dasharray="7 5"/>
 <circle cx="405" cy="309" r="11" fill="#B44F37"/><circle cx="405" cy="165" r="11" fill="#16736E"/>
 <path d="M405 284V190" stroke="#60747A" stroke-width="2" marker-end="url(#arrow)"/>
 <text x="427" y="304" class="body">Fits A, wrong z</text>
 <text x="427" y="150" class="body">Fits A and B</text>
 <rect x="380" y="383" width="50" height="25" rx="4" fill="#16736E"/>
 <path d="M386 379L405 368L424 379" fill="#16736E"/>
 <rect x="62" y="146" width="25" height="40" rx="4" fill="#B44F37"/>
 <path d="M91 150L107 165L91 182" fill="#B44F37"/>
 <text x="238" y="435" class="small">Camera A observes x; z can vary.</text>
 <text x="24" y="113" class="small">Camera B</text><text x="24" y="134" class="small">observes z.</text>
 <text x="145" y="338" class="small">y is also observed by both cameras.</text>`);

save('density.svg',1136,228,'Clone, split and prune are distinct capacity operations',`
 <path d="M378 20V207M756 20V207" stroke="#D6DEDC"/>
 <text x="24" y="35" class="label">CLONE SMALL, HIGH-GRADIENT</text>
 <ellipse cx="83" cy="111" rx="24" ry="16" transform="rotate(-25 83 111)" fill="#16736E" fill-opacity=".3" stroke="#16736E" stroke-width="2"/>
 <path d="M127 111H195" class="line" marker-end="url(#arrow)"/>
 <ellipse cx="251" cy="102" rx="24" ry="16" transform="rotate(-25 251 102)" fill="#16736E" fill-opacity=".2" stroke="#16736E" stroke-width="2"/>
 <ellipse cx="270" cy="119" rx="24" ry="16" transform="rotate(-25 270 119)" fill="#16736E" fill-opacity=".3" stroke="#16736E" stroke-width="2"/>
 <text x="24" y="180" class="body">More adjustable primitives</text><text x="24" y="207" class="small">Shown separated for clarity.</text>
 <text x="402" y="35" class="label">SPLIT LARGE, HIGH-GRADIENT</text>
 <ellipse cx="457" cy="111" rx="42" ry="27" transform="rotate(-25 457 111)" fill="#B44F37" fill-opacity=".2" stroke="#B44F37" stroke-width="2"/>
 <path d="M510 111H571" class="line" marker-end="url(#arrow)"/>
 <ellipse cx="619" cy="123" rx="27" ry="17" transform="rotate(-25 619 123)" fill="#B44F37" fill-opacity=".2" stroke="#B44F37" stroke-width="2"/>
 <ellipse cx="661" cy="97" rx="27" ry="17" transform="rotate(-25 661 97)" fill="#B44F37" fill-opacity=".2" stroke="#B44F37" stroke-width="2"/>
 <text x="402" y="180" class="body">Smaller, more local footprints</text><text x="402" y="207" class="small">The parent is replaced.</text>
 <text x="780" y="35" class="label">PRUNE LOW-OPACITY</text>
 <ellipse cx="842" cy="111" rx="32" ry="23" transform="rotate(-25 842 111)" fill="#60747A" fill-opacity=".08" stroke="#9EAFAF" stroke-width="1.5" stroke-dasharray="5 4"/>
 <path d="M892 111H955" class="line" marker-end="url(#arrow)"/>
 <path d="M995 92L1031 128M1031 92L995 128" stroke="#60747A" stroke-width="2"/>
 <text x="780" y="180" class="body">Remove weak contributions</text><text x="780" y="207" class="small">Size criteria can also trigger removal.</text>`);
console.log('Built four explanatory SVG diagrams.');
