export default function Limits() {
  return (
    <>
<section className="band" id="limits"><div className="shell">
  <div className="head-block">
    <span className="label">Limits</span>
    <h2>Where it stops.</h2>
    <p>Every film tool has a line where the automation ends. Most of them do not tell you
    where. Here is ours.</p>
  </div>

  <div className="limits">
    <div className="row">
      <h3>The ball is not tracked</h3>
      <p>Small, fast, and hidden behind a body for most of a possession. It is a genuinely
      harder problem than the players, and a shot chart built on a ball detector that
      loses the ball is worse than no shot chart.</p>
    </div>
    <div className="row">
      <h3>Events are tagged by hand</h3>
      <p>Shots, rebounds and turnovers are entered by a person against the clip. Automatic
      event recognition is a research programme; the geometry on this page is not, which is
      why one is automated here and the other is not.</p>
    </div>
    <div className="row">
      <h3>The court model knows professional floors</h3>
      <p>The per-frame solver works because the keypoint model recognises this kind of
      court. On a community gym it finds almost nothing — measured, not assumed — and the
      hand-calibrated fallback path takes over.</p>
    </div>
  </div>
</div></section>
    </>
  );
}
