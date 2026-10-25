// Request-order regression: delayed cached responses cannot overwrite a newer UI action.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';

const elements = new Map();
const element = () => ({hidden:false,value:'2024-02-15',classList:{add(){},toggle(){}},removeAttribute(){},replaceChildren(){}});
let resolveFetch;
const seen = [];
const context = vm.createContext({console, Date, AbortController, setTimeout, clearTimeout, fetch:()=>new Promise(resolve=>{resolveFetch=resolve;}),
  document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},querySelector(){return element();},querySelectorAll(){return [];},addEventListener(){}}});
vm.runInContext(fs.readFileSync('frontend/script.js','utf8'),context);
context.showUnavailable = data => seen.push(data);
const first = context.loadPredictionData();
context.clearForecast('A newer date or covered-window selection is in progress.');
resolveFetch({ok:true,json:async()=>({status:'insufficient_data'}),clone(){return this;}});
await first;
assert.equal(seen.length,0,'Old response overwrote newer UI state');
const second = context.loadPredictionData();
resolveFetch({ok:true,json:async()=>({status:'insufficient_data'}),clone(){return this;}});
await second;
assert.equal(seen.length,1,'Current response should be rendered');
console.log('Dashboard request-order regression passed.');
