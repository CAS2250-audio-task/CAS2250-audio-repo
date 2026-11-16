import {COMMANDS,pool} from '../dist/core.mjs';

export function wilson(successes,n,z=1.959963984540054){
 if(!Number.isInteger(n)||n<1||!Number.isInteger(successes)||successes<0||successes>n)throw Error('Invalid binomial counts');
 const p=successes/n,d=1+z*z/n,center=(p+z*z/(2*n))/d,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
 return [Math.max(0,center-half),Math.min(1,center+half)];
}
// Abramowitz-Stegun erf approximation; maximum absolute erf error about 1.5e-7.
export function normalCdf(x){
 const sign=x<0?-1:1,u=Math.abs(x)/Math.SQRT2,t=1/(1+0.3275911*u);
 const erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-u*u);
 return (1+sign*erf)/2;
}
export function analyticSurvival(bank,c){
 const samples=pool(bank,c);
 // Same uniform command, uniform within-command recording policy as the game.
 let q=0;
 for(let command=0;command<COMMANDS.length;command++){
  const recordings=samples.filter(s=>s.command===command);
  for(const sample of recordings){
   const budget=c.deadline-sample.duration_ms-sample.compute_ms*c.scale;
   const onTime=budget<=0?0:c.reactionSigma===0?Number(c.reactionMedian<=budget):normalCdf(Math.log(budget/c.reactionMedian)/c.reactionSigma);
   if(sample.prediction===command)q+=onTime/COMMANDS.length/recordings.length;
  }
 }
 q=Math.max(0,Math.min(1,q));
 let survival=0,choose=1;
 for(let failures=0;failures<=Math.min(c.lives-1,c.rounds);failures++){
  if(failures>0)choose=choose*(c.rounds-failures+1)/failures;
  survival+=choose*Math.pow(1-q,failures)*Math.pow(q,c.rounds-failures);
 }
 return {round_success_probability:q,mission_survival_probability:Math.max(0,Math.min(1,survival)),method:'finite-bank weighted lognormal CDF; binomial P(failures < lives); CDF approximation'};
}
export function summarize(result,bank,config,scenario){
 const s=result.summary,interval=wilson(result.clears,result.games),theory=analyticSurvival(bank,config);
 return {scenario,noise:config.noise,snr:config.noise==='original'?null:config.snr,...config,
  snr:config.noise==='original'?null:config.snr,games:result.games,clears:result.clears,explosions:result.games-result.clears,
  survival_rate:result.clearRate,survival_mc95_low:interval[0],survival_mc95_high:interval[1],
  analytic_survival:theory.mission_survival_probability,analytic_round_success:theory.round_success_probability,
  mean_rounds:result.meanRounds,rounds_observed:s.rounds,rounds_successful:s.successes,
  round_success_rate:s.successes/s.rounds,recognition_matches:s.recognitionCorrect,transmissions:s.attempted,
  recognition_match_rate:s.recognitionCorrect/s.attempted,reaction_median_ms:s.medianReaction,
  recognition_errors:s.flags.recognition,deadline_misses:s.flags.time,choice_errors:s.flags.choice,
  recognition_and_time_errors:result.sessions.reduce((sum,g)=>sum+g.rows.filter(r=>r.recognition_correct===false&&r.deadline_missed).length,0),
  primary_recognition_failures:s.reasons.recognition,primary_time_failures:s.reasons.timeout,
  no_response:s.reasons.no_response,primary_wrong_button:s.reasons.wrong_button};
}
export function toCsv(rows){
 if(!rows.length)return '';
 const keys=Object.keys(rows[0]),cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 return '\uFEFF'+[keys.map(cell).join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n')+'\r\n';
}
