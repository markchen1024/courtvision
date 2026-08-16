export default function Measured() {
  return (
    <>
<section className="band" id="results"><div className="shell">
  <div className="head-block">
    <span className="label">Measured</span>
    <h2>Five possessions, eighty thousand labels, zero wrong.</h2>
    <p>
      The claim on this page is not &ldquo;it works&rdquo; — it is a number, and the number
      was measured. Every possession that ships has a ground truth labelled by hand,
      and every label the renderer draws is replayed against it before the clip goes
      anywhere.
    </p>
  </div>

  <div className="measured">
    <div className="figures">
      <div><div className="n">5</div><div className="k">possessions shipped</div></div>
      <div><div className="n">127 s</div><div className="k">of film, end to end</div></div>
      <div><div className="n">81,000+</div><div className="k">labels checked against hand truth</div></div>
      <div><div className="n">0</div><div className="k">wrong labels</div></div>
    </div>
    <p className="fine">
      A player the pipeline could not name is left unnamed on screen, never
      guessed.
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
