// Synthetic UI fixtures verify data binding, never biological performance.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements=new Map();
function element(){return {textContent:'',hidden:true,children:[],attributes:{},
    replaceChildren(...children){this.children=children;},append(...children){this.children.push(...children);},
    setAttribute(key,value){this.attributes[key]=value;},addEventListener(){}};}
const context=vm.createContext({console,document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element,createElementNS:element,addEventListener(){}}});
vm.runInContext(fs.readFileSync('frontend/hab.js','utf8'),context);
context.testData={input_dates:['2021-01-01','2021-01-02'],freshness:{state:'stale'},historical_demo:true,
    thresholds:{low_threshold:.3,high_threshold:.7},
    zones:[{id:'zone',name:'Test zone',bounds:[74,9,76,11],data_quality:{observation_status:'VALID OBSERVATION',chlorophyll_mean_input_coverage_percent:20,insufficient_input_features:[]}}],
    daily_predictions:Array.from({length:5},(_,i)=>({day:i+1,date:`2021-01-0${i+3}`,zones:[{zone_id:'zone',probability:i===4?null:.8,risk_level:i===4?'UNAVAILABLE':'HIGH'}]}))};
vm.runInContext('habForecast=testData;habRender(1);',context);
assert.match(elements.get('hab-context').textContent,/Day \+1/);
assert.equal(elements.get('hab-rows').children[0].children[1].textContent,'80.0% (experimental)');
vm.runInContext('habRender(5);habDetail(testData.zones[0]);',context);
assert.equal(elements.get('hab-rows').children[0].children[1].textContent,'Unavailable');
assert.match(elements.get('hab-zone-detail').textContent,/20.0%/);
const csv=context.habCSV(context.testData);
assert.match(csv,/operational_warning_available/);assert.match(csv,/"2021-01-07","5"/);
vm.runInContext('habClear()',context);
assert.equal(elements.get('hab-results').hidden,true);
assert.equal(elements.get('hab-map').children.length,0);
console.log('HAB five-day binding, unknown scores, geographic zones, coverage, CSV and clearing passed.');
