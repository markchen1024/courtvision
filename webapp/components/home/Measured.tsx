export default function Measured() {
  return (
    <>
<section className="band" id="results"><div className="shell">
  <div className="head-block">
    <span className="label">Measured</span>
    <h2>Five possessions, eighty thousand labels, zero wrong.</h2>
    <p>
      The claim on this page is not &ldquo;it works&rdquo; — it is a number, and the number
      was measured. Each possession has a per-track ground truth labelled by
      hand; <code>pipeline/score.py</code> replays the renderer&rsquo;s own drawing rule against
      it. Precision counts drawn labels that are right. Coverage counts the ten men on
      court the pipeline could name. A player it could not name stays anonymous on
      screen — blank over wrong, every time.
    </p>
  </div>

  <div className="measured">
    <div className="scroll-x">
      <table className="stat">
        <thead>
          <tr><th>possession</th><th>precision</th><th>coverage</th><th>wrong labels</th></tr>
        </thead>
        <tbody>
          <tr><td className="who">42.6 s</td><td>100.0%</td><td>98.8%</td><td>0</td></tr>
          <tr><td className="who">35.5 s</td><td>100.0%</td><td>96.1%</td><td>0</td></tr>
          <tr><td className="who">19.2 s</td><td>100.0%</td><td>91.7%</td><td>0</td></tr>
          <tr><td className="who">12.5 s</td><td>100.0%</td><td>100.0%</td><td>0</td></tr>
          <tr><td className="who">10.0 s</td><td>100.0%</td><td>99.8%</td><td>0</td></tr>
        </tbody>
      </table>
    </div>
    <p className="fine">
      Coverage below 100% is always a player left unnamed, never a player named
      wrongly.
    </p>

    <div className="reel">
      <video controls playsInline preload="metadata" poster="/media/reel_poster.jpg">
        <source src="/media/reel.mp4" type="video/mp4" />
      </video>
      <p className="fine">
        All five possessions, cut together. The card before each clip carries its
        measured numbers — they come from the evaluation set, not from enthusiasm.
      </p>
    </div>
  </div>
</div></section>
    </>
  );
}
