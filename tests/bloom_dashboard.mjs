// Software fixture or real inference JSON; never interpreted as biological truth.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements=new Map();
function element(){return {hidden:false,textContent:'',children:[],replaceChildren(...nodes){this.children=nodes;},append(node){this.children.push(node);},removeAttribute(key){delete this[key];},addEventListener(){}};}
const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
const context=vm.createContext({console,encodeURIComponent,document:{getElementById:get,createElement:element,addEventListener(){}}});
vm.runInContext(fs.readFileSync('frontend/bloom.js','utf8'),context);
const data=process.argv[2]?JSON.parse(fs.readFileSync(process.argv[2],'utf8')):{
    forecast_id:'a'.repeat(32),input_dates:['2024-02-11','2024-02-12','2024-02-13','2024-02-14','2024-02-15'],
    historical_demo:true,threshold:.5,data_quality:{chlorophyll_mean_input_coverage_percent:10},freshness:{state:'stale'},validation_limit:'Test fixture only.',
    maps:Array.from({length:5},(_,i)=>`/bloom-forecasts/${'a'.repeat(32)}/day_${i+1}.png`),
    daily_predictions:Array.from({length:5},(_,i)=>({day:i+1,date:`2024-02-${16+i}`,max_score:.1,event_pixels:0,supported_pixels:100}))};
context.actualData=data;vm.runInContext('bloomRender(actualData)',context);
for(let day=1;day<=5;day++){
    context.bloomDay(day);assert.equal(get('bloom-map').src,data.maps[day-1]);assert.match(get('bloom-context').textContent,new RegExp(`Day \\+${day}`));
}
assert.equal(get('bloom-table').children.length,5);
assert.equal(get('bloom-export').href,`/api/bloom/forecasts/${data.forecast_id}/summary.csv?demo=true`);
assert.match(get('bloom-coverage').textContent,/coverage/);
context.bloomClear();assert.equal(get('bloom-results').hidden,true);assert.equal(get('bloom-map').src,undefined);
console.log('Satellite bloom-event five-day binding, maps, coverage, CSV routing and clearing passed.');
// Switching experiments while status is in flight must discard the old response.
const pending=[];
context.AbortSignal=AbortSignal;
context.fetch=url=>new Promise(resolve=>pending.push({url,resolve}));
get('bloom-run').value='run_01';
const strict=context.bloomStatus();
get('bloom-run').value='run_02';
const relaxed=context.bloomStatus();
pending[1].resolve({ok:true,json:async()=>({run_id:'run_02',status:'ready_research_only',message:'Relaxed ready',label_definition:'P75'})});
await relaxed;
pending[0].resolve({ok:true,json:async()=>({run_id:'run_01',status:'ready_research_only',message:'Strict ready',label_definition:'P90'})});
await strict;
assert.match(get('bloom-status').textContent,/run_02/);
assert.equal(get('bloom-definition').textContent,'P75');
assert.match(get('bloom-evaluation').href,/run_id=run_02/);
const outdated=context.bloomStatus();
pending[2].resolve({ok:true,json:async()=>({status:'ready_research_only',message:'Old server'})});
await outdated;
assert.equal(get('bloom-submit').disabled,true);
assert.match(get('bloom-status').textContent,/server does not support/);
console.log('Experiment switch race and outdated-server identity guard passed.');
const missingAPI=context.bloomStatus();
pending[3].resolve({ok:false,status:404});
await missingAPI;
assert.equal(get('bloom-submit').disabled,true);
assert.match(get('bloom-status').textContent,/Restart.*README.*8020/);
console.log('Missing experiment API provides an actionable server restart instruction.');
