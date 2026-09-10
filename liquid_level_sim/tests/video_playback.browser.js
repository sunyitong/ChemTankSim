// Run in a loaded local/deployed page with: agent-browser eval --stdin < this-file
// Exercises real controls, decoded pixels and media time; no detector hooks or replacement results.
(async () => {
  const q = selector => document.querySelector(selector);
  const wait = async (condition, message, timeout = 12000) => {
    const deadline = performance.now() + timeout;
    while (!condition()) {
      if (performance.now() > deadline) throw new Error(message);
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  };
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  check((window.CASES || []).find(c=>c.id==='F5')?.label.includes('P5'), 'Page still contains the pre-approval F5 bundle');
  const snapshot = () => {
    const video = q('#vid'), canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video, 0, 0);
    const data = ctx.getImageData(0,0,canvas.width,canvas.height).data;
    let hash = 2166136261, colourEnergy = 0, samples = 0;
    for (let i = 0; i < data.length; i += 64) {
      hash = Math.imul(hash ^ data[i], 16777619);
      colourEnergy += Math.abs(data[i]-data[i+1]) + Math.abs(data[i+1]-data[i+2]); samples++;
    }
    return {frame:+q('#vseek').value, actualTime:video.currentTime, shownTime:q('#vtime').textContent,
      pixelHash:hash>>>0, colourEnergy:colourEnergy/(2*samples), level:q('#mLevel').textContent,
      seekable:Array.from({length:video.seekable.length},(_,i)=>[video.seekable.start(i),video.seekable.end(i)])};
  };
  const select = async id => {
    q(`#cases button[data-id="${id}"]`).click();
    await wait(()=>q('#vid').readyState>=2 && q('#vid').currentTime>0 && q('#vtime').textContent==='0.00 s', `${id}: first frame did not decode`);
  };
  const seek = async (i,fps) => {
    const slider=q('#vseek'); slider.value=String(i); slider.dispatchEvent(new Event('input',{bubbles:true}));
    await wait(()=>q('#vtime').textContent===(i/fps).toFixed(2)+' s' && Math.abs(q('#vid').currentTime-(i+.5)/fps)<.001, `Seek ${i}: display and decoded frame disagree`);
    return snapshot();
  };
  const report={url:location.href,cases:{}};
  for (const id of ['F4','F5']) {
    await select(id);
    check(q('#vid').currentSrc.startsWith('blob:'),`${id}: external clip is not locally seekable`);
    const frames=[snapshot(), await seek(50,25), await seek(100,25)];
    check(new Set(frames.map(f=>f.pixelHash)).size===3,`${id}: decoded video pixels are frozen`);
    check(id==='F5' ? frames[1].colourEnergy>5 : frames[1].colourEnergy<2, `${id}: wrong colour/white-backlight asset`);
    // Replaying from the end must advance the real media clock and finish at the actual last frame.
    q('#vplay').click();
    await wait(()=>+q('#vseek').value>5 && +q('#vseek').value<95,`${id}: replay did not advance`);
    const playing=snapshot();
    check(playing.actualTime>.2 && playing.actualTime<4,`${id}: fake progress without a media seek`);
    await wait(()=>q('#vplay').getAttribute('aria-label')==='Play' && +q('#vseek').value===100,`${id}: playback did not finish`);
    const end=snapshot();
    check(Math.abs(end.actualTime-4.02)<.001,`${id}: end frame not decoded`);
    report.cases[id]={frames,playing,end};
  }
  // Existing embedded sequence remains compatible, and rapid case changes cancel the old load.
  await select('F1');
  report.cases.F1={first:snapshot(),middle:await seek(40,20)};
  check(report.cases.F1.first.pixelHash!==report.cases.F1.middle.pixelHash,'Legacy F1 is frozen');
  q('#cases button[data-id="F4"]').click(); q('#cases button[data-id="F5"]').click();
  await wait(()=>q('#vid').readyState>=2 && q('#vid').currentTime>0 && q('#sampleChip').textContent.startsWith('F5'), 'Rapid switch did not load F5');
  report.rapidSwitch=await seek(75,25);
  report.passed=true;
  return report;
})()
