export const COMMANDS=['출발','정지','전진','후진','왼쪽','오른쪽','시작','복귀','대기','재시작'];
export const SYMBOLS=['↗','■','↑','↓','←','→','▶','↩','Ⅱ','⟳'];
export const DEFAULT_CONFIG=Object.freeze({noise:'restaurant',snr:0,deadline:3200,scale:100,lives:3,rounds:20,reactionMedian:450,reactionSigma:0.35,seed:42});
export function checkedConfig(input){
 const c={...DEFAULT_CONFIG,...input};
 if(!['original','restaurant','car','compressor'].includes(c.noise))throw Error('음향 환경을 확인하세요.');
 for(const k of ['deadline','scale','lives','rounds','reactionMedian','reactionSigma','seed']){
  if(!Number.isFinite(c[k]))throw Error('설정은 유한한 숫자여야 합니다: '+k);
 }
 if(c.noise!=='original'&&![10,0,-5].includes(c.snr))throw Error('SNR 설정을 확인하세요.');
 if(c.deadline<=0||c.scale<1||!Number.isInteger(c.lives)||c.lives<1||!Number.isInteger(c.rounds)||c.rounds<1||c.rounds>1000||c.reactionMedian<=0||c.reactionSigma<0||!Number.isInteger(c.seed)||c.seed<0||c.seed>2147483647)throw Error('미션 설정 범위를 확인하세요.');
 return c;
}
export function validateBank(bank){
 if(!bank||!Array.isArray(bank.samples)||!bank.samples.length||COMMANDS.some((v,i)=>bank.labels?.[i]!==v))throw Error('음성 데이터 형식이 맞지 않습니다.');
 const ids=new Set();
 for(const s of bank.samples){
  if(ids.has(s.id)||typeof s.id!=='string'||!Number.isInteger(s.command)||s.command<0||s.command>=10||!Number.isInteger(s.prediction)||s.prediction<0||s.prediction>=bank.labels.length||!Number.isFinite(s.compute_ms)||s.compute_ms<0||!Number.isFinite(s.duration_ms)||s.duration_ms<=0||!/^audio\/[a-f0-9]+\.wav$/.test(s.audio))throw Error('손상된 음성 데이터가 있습니다.');
  ids.add(s.id);
 }
 for(const noise of ['original','restaurant','car','compressor'])for(const snr of noise==='original'?[null]:[10,0,-5])for(let command=0;command<10;command++){
  if(!bank.samples.some(s=>s.noise===noise&&(noise==='original'||s.snr===snr)&&s.command===command))throw Error('누락된 음성 조건이 있습니다.');
 }
 return bank;
}
export const SCENARIOS=[
 ['출격 승인','격납고가 열렸습니다. 수송 로봇을 출발시키세요.','GATE OPEN'],
 ['충돌 경고','진행 경로에 잔해가 있습니다. 즉시 정지하세요.','COLLISION'],
 ['통로 확보','전방 통로가 열렸습니다. 앞으로 전진하세요.','CLEAR PATH'],
 ['거리 확보','압력파가 접근합니다. 뒤로 후진하세요.','PRESSURE WAVE'],
 ['왼쪽 회피','오른쪽에서 소행성이 접근합니다. 왼쪽으로 피하세요.','EVADE LEFT'],
 ['오른쪽 회피','왼쪽에서 소행성이 접근합니다. 오른쪽으로 피하세요.','EVADE RIGHT'],
 ['작업 개시','구조 장치가 준비됐습니다. 작업을 시작하세요.','SYSTEM READY'],
 ['기지 복귀','화물 적재 완료. 안전 구역으로 복귀하세요.','RETURN HOME'],
 ['신호 대기','교차 구역에 다른 기체가 있습니다. 대기하세요.','HOLD POSITION'],
 ['시스템 복구','제어기가 멈췄습니다. 시스템을 재시작하세요.','SYSTEM OFFLINE']
];
export function rng(seed=42){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export function normal(random){const u=Math.max(Number.MIN_VALUE,random()),v=random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
export function reactionMs(median,sigma,random){return median*Math.exp(sigma*normal(random));}
export function pool(bank,c){return bank.samples.filter(s=>s.noise===c.noise&&(c.noise==='original'||s.snr===Number(c.snr)));}
export function pickSample(samples,command,random){const p=samples.filter(s=>s.command===command);if(!p.length)throw Error('해당 명령어의 실제 음성 묶음이 없습니다.');return p[Math.floor(random()*p.length)];}
export function evaluate({target,pressed,sample,reaction,config}){
 const compute=sample?sample.compute_ms*config.scale:0;
 const audio=sample?sample.duration_ms:0;
 const total=reaction==null?null:reaction+audio+compute;
 const choice=pressed===target;
 const recognition=sample?sample.prediction===pressed:null;
 const late=total==null||total>config.deadline;
 const success=choice&&recognition&&!late;
 const reason=pressed==null?'no_response':!choice?'wrong_button':late?'timeout':!recognition?'recognition':'success';
 return {target,pressed,prediction:sample?.prediction??null,sample_id:sample?.id??null,source_id:sample?.source_id??null,
   reaction_ms:reaction,audio_ms:audio,measured_compute_ms:sample?.compute_ms??null,
   displayed_compute_ms:compute,total_ms:total,choice_correct:choice,recognition_correct:recognition,
   deadline_missed:late,success,reason,noise:config.noise,snr:config.noise==='original'?null:config.snr,
   deadline_ms:config.deadline,latency_scale:config.scale};
}
export function stats(rows){
 const done=rows.filter(r=>r.reason!=='aborted'),attempted=done.filter(r=>r.pressed!==null),rt=attempted.map(r=>r.reaction_ms).sort((a,b)=>a-b);
 const median=rt.length?(rt[Math.floor((rt.length-1)/2)]+rt[Math.ceil((rt.length-1)/2)])/2:null;
 return {rounds:done.length,attempted:attempted.length,successes:done.filter(r=>r.success).length,
   recognitionCorrect:attempted.filter(r=>r.recognition_correct).length,medianReaction:median,
   flags:{recognition:attempted.filter(r=>r.recognition_correct===false).length,
     time:done.filter(r=>r.deadline_missed).length,
     choice:done.filter(r=>!r.choice_correct).length},
   reasons:Object.fromEntries(['success','wrong_button','recognition','timeout','no_response'].map(k=>[k,done.filter(r=>r.reason===k).length]))};
}
export function simulate(bank,config,games=1000){
 config=checkedConfig(config);
 if(!Number.isInteger(games)||games<1||games>10000)throw Error('시뮬레이션 횟수 범위를 확인하세요.');
 const samples=pool(bank,config),all=[];let clears=0,totalRounds=0;
 for(let game=0;game<games;game++){
  const taskRandom=rng(config.seed+game*13),sampleRandom=rng(config.seed+100000+game*13),reactionRandom=rng(config.seed+200000+game*13);
  let strikes=0;const rows=[];
  for(let round=0;round<config.rounds;round++){
   const target=Math.floor(taskRandom()*10),sample=pickSample(samples,target,sampleRandom);
   const reaction=reactionMs(config.reactionMedian,config.reactionSigma,reactionRandom);
   // Automated pilot always chooses the requested command. Sample identity and
   // latency are drawn without consulting its prediction or outcome.
   const row=evaluate({target,pressed:target,sample,reaction,config});row.game=game;row.round=round+1;rows.push(row);
   if(!row.success)strikes++;if(strikes>=config.lives)break;
  }
  const survived=strikes<config.lives;if(survived)clears++;
  totalRounds+=rows.length;all.push({game,survived,strikes,rows});
 }
 return {games,clears,clearRate:clears/games,meanRounds:totalRounds/games,summary:stats(all.flatMap(g=>g.rows)),sessions:all};
}
export function missionStreams(seed,game=0){return {task:rng(seed+game*13),sample:rng(seed+100000+game*13),reaction:rng(seed+200000+game*13)};}
export function csvText(records){
 const keys=['record_type','session_id','mode','session_status','game','round','target_label','pressed_label','prediction_label','target','pressed','prediction','reaction_ms','audio_ms','measured_compute_ms','displayed_compute_ms','total_ms','choice_correct','recognition_correct','deadline_missed','success','reason','noise','snr','deadline_ms','latency_scale','sample_id','source_id','audio','waveform_sha256','seed','lives','mission_rounds','reaction_median_ms','reaction_sigma','reaction_source','bank_version','bank_sha256','speaker','speaker_role','detector_connected','inference'];
 const escape=v=>{const t=v==null?'':String(v);return '"'+t.replaceAll('"','""')+'"';};
 return '\uFEFF'+[keys.map(escape).join(','),...records.map(r=>keys.map(k=>escape(r[k])).join(','))].join('\r\n');
}
