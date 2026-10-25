// UI contract fixture (not scientific observations), or optionally real backend JSON.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const data = process.argv[2] ? JSON.parse(fs.readFileSync(process.argv[2], 'utf8')) : {
  status:'success',forecast_id:'a'.repeat(32),days:5,height:120,width:84,
  input_dates:['2024-02-11','2024-02-12','2024-02-13','2024-02-14','2024-02-15'],
  forecast_dates:Array.from({length:5},(_,i)=>`2024-02-${16+i}`),
  maps:Array.from({length:5},(_,i)=>`/forecasts/${'a'.repeat(32)}/day_${i+1}.png`),
  historical_demo:true,operational_warnings_available:false,generated_at_utc:'2024-02-15T12:00:00Z',
  freshness:{state:'stale',age_days:100},data_quality:{state:'observed',chlorophyll_mean_input_coverage_percent:10},
  decision_rule:{type:'argmax'},warning_parameters:{regional_elevated_percent:10,regional_strong_percent:20},
  current_observed_risk:{warning_level:'NORMAL',valid_pixels:4,coverage_percent:100},environment:[],
  daily_predictions:Array.from({length:5},(_,i)=>({day:i+1,warning_level:'NORMAL',supported_pixels:4,
    elevated_risk_pixels:0,high_risk_pixels:0,high_risk_percentage:0,max_probability:.2,max_elevated_probability:.1}))
};
const elements = new Map();
function element() {
  return {textContent:'', value:'', disabled:false, hidden:false, listeners:{}, style:{},
    classList:{add(){},remove(){},toggle(){}},
    addEventListener(name,handler){this.listeners[name]=handler;},removeAttribute(name){delete this[name];},
    replaceChildren(...children){this.children=children;},append(){},insertRow(){return {append(){},insertCell(){return {};}};}};
}
function get(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);}
const days=Array.from({length:5},element);
let initialize, resolvePrediction;
const requests=[];
const response=value=>({ok:true,json:async()=>value,clone(){return response(value);}});
const context=vm.createContext({console,Date,encodeURIComponent,AbortController,setTimeout,clearTimeout,setInterval(){},
  document:{getElementById:get,querySelector:element,querySelectorAll:()=>days,createElement:element,
    addEventListener(name,handler){if(name==='DOMContentLoaded')initialize=handler;}},
  fetch:async(url,options={})=>{
    if(url==='/api/health')return response({latest_data_date:'2025-12-31'});
    if(url==='/api/system')return response({live_update:{state:'not_configured'},satellite_sources:{active:'MODIS',sentinel3:{state:'unavailable'}},model:{version:'frozen'},calibration:{}});
    if(url==='/api/live/prediction')return response({status:'unavailable'});
    if(url==='/api/prediction?demo=false')return response({status:'stale'});
    if(url==='/api/covered-dates')return response({dates:[{input_end_date:'2024-02-15',mean_chlorophyll_coverage_percent:10,chlorophyll_observed_days:3},{input_end_date:'2024-12-31',mean_chlorophyll_coverage_percent:20,chlorophyll_observed_days:2}]});
    if(url==='/api/predict'){
      requests.push(JSON.parse(options.body));
      return new Promise(resolve=>{resolvePrediction=resolve;});
    }
    if(url==='/slow')return new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(Object.assign(new Error(),{name:'AbortError'}))));
    throw new Error(`Unexpected request: ${url}`);
  }});
vm.runInContext(fs.readFileSync('frontend/script.js','utf8'),context);
initialize();
await new Promise(setImmediate);
assert.ok(days.every(button=>button.disabled),'Day controls must be disabled without a forecast');
assert.match(get('data-notice').textContent,/Load historical example/);
const example=get('covered-date');
const action=example.listeners.click({currentTarget:example});
await new Promise(setImmediate);
assert.deepEqual(requests,[{input_end_date:'2024-02-15',historical_demo:true}]);
assert.equal(get('historical-demo').checked,true);
assert.equal(get('generate-forecast').textContent,'Generating…');
assert.match(get('forecast-progress').textContent,/Running the model/);
assert.ok(days.every(button=>button.disabled));
resolvePrediction(response(data));
await action;
assert.ok(days.every(button=>!button.disabled),'Loaded forecast must enable day switching');
assert.equal(get('generate-forecast').textContent,'Generate forecast');
assert.equal(example.textContent,'Load historical example');
assert.equal(get('forecast-progress').textContent,'');
context.showDay(5);
assert.equal(get('prediction-map').src,data.maps[4]);
await assert.rejects(context.fetchWithTimeout('/slow',{},5),/taking too long/);
assert.match(context.forecastError('Input data are outdated; request historical_demo'),/Load historical example/);
await get('list-covered-dates').listeners.click();
assert.equal(get('eligible-input-date').children.length,3);
assert.match(get('eligible-input-date').children[1].textContent,/10.0% coverage/);
get('eligible-input-date').listeners.change({target:{value:'2024-12-31'}});
assert.equal(get('input-end-date').value,'2024-12-31');
assert.equal(get('historical-demo').checked,true);
assert.match(get('data-notice').textContent,/Click Generate forecast/);
assert.ok(days.every(button=>button.disabled));
console.log('One-click example, eligible-date selector, stale guidance, busy feedback, day switching and timeout passed.');
