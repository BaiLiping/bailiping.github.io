function revealTarget(){const id=decodeURIComponent(location.hash.slice(1));if(!id)return;const target=document.getElementById(id);if(!target)return;const detail=target.closest('details');if(detail)detail.open=true;}
window.addEventListener('hashchange',revealTarget);revealTarget();
