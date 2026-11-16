#!/usr/bin/env node
import {readFileSync,writeFileSync,mkdirSync,appendFileSync,existsSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {DEFAULT_CONFIG,COMMANDS,checkedConfig,validateBank,simulate,pool} from '../dist/core.mjs';
import {summarize,toCsv} from './statistics.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const options={games:10000,seed:42,deadline:3200,scale:100,lives:3,rounds:20,median:450,sigma:.35,out:path.join(root,'reports/game-statistics'),plots:true,logs:false};
const usage=`Usage: node analysis/run-stats.mjs [options]
 --games 10000       games per acoustic condition (1..10000)
 --seed 42          deterministic seed
 --deadline 3200    base deadline in ms
 --scale 100        measured compute latency multiplier
 --lives 3 --rounds 20 --median 450 --sigma 0.35
 --out PATH         NEW or EMPTY output directory
 --round-logs       include all base-scenario round rows in CSV (large)
 --no-plots         skip Python/matplotlib PNG/PDF generation
Sweeps: original + 3 noises x 3 SNRs; deadlines 2200/2700/3200/3700/4200ms;
unscaled-compute comparison. Reuses existing core.mjs; no new inference/training.`;
const args=process.argv.slice(2);
try{
 for(let i=0;i<args.length;i++){
  const key=args[i];if(key==='--help'){console.log(usage);process.exit(0);}if(key==='--no-plots'){options.plots=false;continue;}if(key==='--round-logs'){options.logs=true;continue;}
  const name=key.slice(2);if(!key.startsWith('--')||!Object.hasOwn(options,name)||['plots','logs'].includes(name)||args[i+1]===undefined)throw Error('Unknown or incomplete option: '+key);
  options[name]=name==='out'?path.resolve(args[++i]):Number(args[++i]);
 }
 if(!Number.isInteger(options.games)||options.games<1||options.games>10000)throw Error('--games must be 1..10000');
 const base=checkedConfig({...DEFAULT_CONFIG,seed:options.seed,deadline:options.deadline,scale:options.scale,lives:options.lives,rounds:options.rounds,reactionMedian:options.median,reactionSigma:options.sigma});
 if(existsSync(options.out)&&readdirSync(options.out).length)throw Error('Output directory is not empty; choose a new --out to preserve prior results.');
 mkdirSync(options.out,{recursive:true});
 const bankBytes=readFileSync(path.join(root,'dist/bank.json')),bank=validateBank(JSON.parse(bankBytes));
 const sha=b=>createHash('sha256').update(b).digest('hex');
 const conditions=[{noise:'original',snr:null},...['car','restaurant','compressor'].flatMap(noise=>[10,0,-5].map(snr=>({noise,snr})))];
 const scenarios=[{id:'base',deadline:base.deadline,scale:base.scale},...[2200,2700,3200,3700,4200].filter(d=>d!==base.deadline).map(deadline=>({id:'deadline_sweep',deadline,scale:base.scale})),...(base.scale===1?[]:[{id:'unscaled_compute',deadline:base.deadline,scale:1}])];
 const summary=[],gamePath=path.join(options.out,'games.csv'),roundPath=path.join(options.out,'rounds-base.csv');let gameHeader=false,roundHeader=false;
 for(const scenario of scenarios)for(const condition of conditions){
  const c=checkedConfig({...base,...condition,deadline:scenario.deadline,scale:scenario.scale}),result=simulate(bank,c,options.games);
  const row=summarize(result,bank,c,scenario.id);summary.push(row);
  const games=result.sessions.map(g=>({scenario:scenario.id,noise:c.noise,snr:c.noise==='original'?null:c.snr,deadline_ms:c.deadline,latency_scale:c.scale,seed:c.seed,game:g.game,survived:g.survived,strikes:g.strikes,rounds:g.rows.length,successes:g.rows.filter(r=>r.success).length,recognition_errors:g.rows.filter(r=>r.recognition_correct===false).length,deadline_misses:g.rows.filter(r=>r.deadline_missed).length}));
  appendCsv(gamePath,games,gameHeader);gameHeader=true;
  if(options.logs&&scenario.id==='base'){
   const rows=result.sessions.flatMap(g=>g.rows.map(r=>({...r,seed:c.seed,scenario:scenario.id})));
   appendCsv(roundPath,rows,roundHeader);roundHeader=true;
  }
  console.log(`${scenario.id.padEnd(16)} ${c.noise.padEnd(10)} ${String(c.snr??'—').padStart(3)} dB / ${c.deadline} ms / x${c.scale}: ${result.clears}/${result.games} survived (${(result.clearRate*100).toFixed(2)}%)`);
 }
 const bankConditions=conditions.map(c=>{const samples=pool(bank,c);return {...c,unique_sources:new Set(samples.map(s=>s.source_id)).size,n:samples.length,recognition_matches:samples.filter(s=>s.command===s.prediction).length,recognition_rate:samples.filter(s=>s.command===s.prediction).length/samples.length};});
 const commandRows=conditions.flatMap(c=>COMMANDS.map((command,i)=>{const samples=pool(bank,c).filter(s=>s.command===i);return {...c,command,n:samples.length,recognition_matches:samples.filter(s=>s.prediction===i).length};}));
 const {samples,...provenance}=bank;
 const manifest={created_at:new Date().toISOString(),options,base_config:base,scenarios,provenance,bank_sha256:sha(bankBytes),code_hashes:Object.fromEntries(['dist/core.mjs','analysis/statistics.mjs','analysis/run-stats.mjs','analysis/plot_stats.py'].map(p=>[p,sha(readFileSync(path.join(root,p)))])),node:process.version,
  interpretation:{inference:'precomputed actual predictions, not live inference',speaker:'spk03 development validation; no independent test',detector_connected:false,reaction:'assumed lognormal; no empirical human calibration; always correct button',ci:'95% Wilson binomial interval for Monte Carlo mission survival conditional on this fixed bank and assumptions; not model generalization uncertainty',comparison:'same seeds across conditions; common random numbers; condition estimates are paired, not independent',early_stopping:'stop on lives-th failure, including failure on last round; alive after all rounds is survival',round_denominators:'observed transmitted rounds, including terminal failure; no unplayed rounds; error flags overlap',timing:'reaction + 1500ms audio + measured compute * scale; excludes IO/network/output-device/browser delays',analytic:'finite-bank weighted lognormal CDF and binomial survival, approximate normal CDF; cross-check only'}};
 writeFileSync(path.join(options.out,'summary.json'),JSON.stringify({manifest,summary,bank_conditions:bankConditions},null,2)+'\n');
 writeFileSync(path.join(options.out,'summary.csv'),toCsv(summary));writeFileSync(path.join(options.out,'bank-conditions.csv'),toCsv(bankConditions));writeFileSync(path.join(options.out,'bank-commands.csv'),toCsv(commandRows));
 writeFileSync(path.join(options.out,'REPORT_KO.md'),report(summary,bankConditions,manifest));
 if(options.plots){const p=spawnSync('python',[path.join(root,'analysis/plot_stats.py'),options.out],{stdio:'inherit'});if(p.status!==0)throw Error('Statistics saved, but plot generation failed. Run analysis/plot_stats.py separately.');}
 console.log('Saved: '+options.out);
}catch(error){console.error(error.message);process.exitCode=1;}

function appendCsv(file,rows,hasHeader){let csv=toCsv(rows);if(hasHeader)csv=csv.slice(csv.indexOf('\r\n')+2);appendFileSync(file,csv);}
function report(rows,bankConditions,m){
 const p=v=>(100*v).toFixed(2),names={original:'원본(추가 소음 없음)',car:'차량',restaurant:'식당',compressor:'압축기'},label=r=>names[r.noise]+(r.noise==='original'?'':` ${r.snr} dB`),base=rows.filter(r=>r.scenario==='base');
 return `# COMMAND RESCUE: 로봇 무사귀환 보고서\n\n`+
 `## 방법\n\n기존 게임의 core.mjs simulate/evaluate를 직접 불러 자동 실행했다. 음향 10조건에서 각각 ${m.options.games.toLocaleString()}게임을 반복했다. 1게임은 최대 ${m.base_config.rounds}라운드이며 실패 ${m.base_config.lives}회에 도달하면 즉시 종료한다. 기본 제한시간은 ${m.base_config.deadline}ms, 처리 지연 배율은 ×${m.base_config.scale}, seed는 ${m.base_config.seed}이다. 반응시간은 중앙값 ${m.base_config.reactionMedian}ms, 로그 표준편차 σ=${m.base_config.reactionSigma}인 로그정규 가정 분포로 표집하며 자동 조종은 항상 요구 버튼을 선택한다.\n\n`+
 `명령은 10종 균등 표집, 해당 명령의 녹음 3개 중 균등 복원 표집한다. 판정은 올바른 버튼 ∧ 올바른 인식 ∧ 반응시간+녹음 길이+실측 처리시간×배율 ≤ 제한시간이다. 모든 조건에서 같은 seed를 사용하므로 조건 간 비교는 공통 난수에 의한 짝지어진 비교다.\n\n`+
 `## 기본 조건 결과\n\n| 음향 조건 | 귀환/출격 | 무사귀환율 % (95% MC 구간) | 평균 진행 라운드 | 전송 인식 일치율 % | 시간 초과/전송 |\n|---|---:|---:|---:|---:|---:|\n`+
 base.map(r=>`| ${label(r)} | ${r.clears}/${r.games} | ${p(r.survival_rate)} (${p(r.survival_mc95_low)}–${p(r.survival_mc95_high)}) | ${r.mean_rounds.toFixed(2)} | ${p(r.recognition_match_rate)} | ${r.deadline_misses}/${r.transmissions} |`).join('\n')+
 `\n\n무사귀환은 모든 라운드 종료 시 누적 실패가 ${m.base_config.lives}회 미만인 경우이다. 게임이 일찍 종료되므로 라운드 통계의 분모는 실제 진행한 전송 횟수이며, 최대 라운드 수×출격 수를 사용하지 않는다. 인식 오류와 시간 초과는 동시에 발생할 수 있다. summary.csv의 primary_*는 대표 사유, *_errors와 deadline_misses는 독립 플래그다.\n\n`+
 `## 보고서에 붙일 설명\n\n본 시뮬레이션은 개발 검증 화자 spk03의 30개 원본 녹음으로 만든 300개 고정 음성 조건의 실제 사전 추론 결과를 재표집하였다. 각 음향 조건의 bank 자체 인식 일치율은 bank-conditions.csv(n=30), 명령별 결과는 bank-commands.csv(n=3)에 별도로 제공한다. 전송 인식 일치율은 게임에서 표집된 결과의 일치율이며 bank 전체 비율과 구분한다.\n\n`+
 `생존율의 95% 구간은 고정 bank와 가정한 반응시간 분포 아래 반복 시뮬레이션의 Wilson 구간이다. 녹음·화자·모델의 일반화 불확실성을 나타내지 않는다. 반복 수를 늘려도 새로운 음성 평가가 늘어나는 것은 아니다. 반응시간은 사람에게서 실측·보정하지 않았고, 조작 실수는 가정하지 않았다. 따라서 사람의 실제 플레이 성과로 해석할 수 없다.\n\n`+
 `검출기는 미연결이며, 브라우저 실시간 추론·새 독립 test 성능·검출기 포함 전체 방식의 성능이 아니다. SNR은 원본 전체 1.5초 파형 전력 기준이다. 처리 지연 ×${m.base_config.scale}는 시연 배율이며 하드웨어 실측 시간을 그대로 의미하지 않는다. unscaled_compute는 같은 bank의 실측 compute_ms에 ×1만 적용한 비교로, 실제 시스템 종단 지연 측정이 아니다.\n\n`+
 `## 그림\n\n- survival.png / survival.pdf: 음향 조건별 무사귀환율과 95% Monte Carlo 구간. analytic_survival은 같은 가정의 수식 검산값이다.\n- deadline-sweep.png / .pdf: 제한시간에 따른 생존율 민감도.\n- failure-flags.png / .pdf: 진행한 라운드의 인식 오류·시간 초과·교집합 비율. 두 오류 막대는 중복될 수 있다.\n- compute-scale.png / .pdf: 기본 배율과 ×1 비교.\n\n그림의 축은 글꼴 호환성을 위해 영어로 표기했다. PNG는 220dpi, PDF는 벡터 출력이다.\n\n`+
 `## 재현 및 해석 메모\n\nsummary.json에 설정·코드/bank SHA-256·모델·소음 출처를 저장했다. games.csv는 모든 시나리오의 게임별 결과다. 선택한 --round-logs 옵션은 기본 시나리오의 라운드 원시 행도 기록한다. 독립·동일분포 라운드 가정 아래 이항 생존확률을 별도로 계산하여 시뮬레이션과 대조했다.\n\n약간의 게임식 표현: 이 표의 ‘폭발’은 로봇의 퇴근 실패를 뜻한다. 로봇 보험료는 계산하지 않았다.\n`;
}
