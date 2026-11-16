import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {wilson,normalCdf,analyticSurvival,summarize} from '../analysis/statistics.mjs';
import {DEFAULT_CONFIG,simulate} from '../dist/core.mjs';
const bank=JSON.parse(readFileSync(new URL('../dist/bank.json',import.meta.url)));
test('Wilson interval handles zero and all survivors without zero uncertainty',()=>{
 const a=wilson(0,100),b=wilson(100,100);assert.ok(a[0]<1e-15&&a[1]>.036&&a[1]<.038);assert.ok(b[0]>.962&&b[0]<.964&&b[1]>1-1e-15);assert.throws(()=>wilson(1,0));
});
test('normal CDF and deterministic reaction boundary',()=>{
 assert.ok(Math.abs(normalCdf(0)-.5)<1e-7);assert.ok(Math.abs(normalCdf(1.96)-.975)<1e-5);
 const toy={samples:Array.from({length:10},(_,command)=>({noise:'original',command,prediction:command,compute_ms:0,duration_ms:1500}))};
 const c={...DEFAULT_CONFIG,noise:'original',reactionSigma:0,deadline:1950};
 assert.equal(analyticSurvival(toy,c).mission_survival_probability,1);
 assert.equal(analyticSurvival(toy,{...c,deadline:1949}).mission_survival_probability,0);
});
test('game summary matches core counts and analytic probability within Monte Carlo error',()=>{
 for(const noise of ['original','car','restaurant','compressor']){
  const c={...DEFAULT_CONFIG,noise},result=simulate(bank,c,10000),s=summarize(result,bank,c,'base');
  assert.equal(s.transmissions,result.summary.attempted);assert.equal(s.clears+s.explosions,10000);
  assert.ok(Math.abs(s.survival_rate-s.analytic_survival)<6*Math.sqrt(s.analytic_survival*(1-s.analytic_survival)/10000)+.001);
  assert.ok(s.recognition_and_time_errors<=Math.min(s.recognition_errors,s.deadline_misses));
 }
});
