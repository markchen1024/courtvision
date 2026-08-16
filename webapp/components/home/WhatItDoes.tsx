export default function WhatItDoes() {
  return (
    <>
<section className="band"><div className="shell">
  <div className="head-block">
    <span className="label">What it does</span>
    <h2>Pixels are not basketball. Metres are.</h2>
    <p>
      Detection tells you there is a person at the bottom-left of the frame. That is not a
      statistic. The transform from camera to floor is the step that makes everything
      downstream mean something — and it is the part most film tools skip.
    </p>
  </div>

  <div className="grid-3">
    <div className="cell lead">
      <span className="mark">H</span>
      <h3>Court homography</h3>
      <p>Court markings in the frame are matched to their known positions on a standard
      plan, and <code>cv2.findHomography</code> solves the 3×3 that relates them. Positions
      come out in metres, so distances are distances.</p>
    </div>
    <div className="cell">
      <span className="mark">D</span>
      <h3>Detection &amp; tracking</h3>
      <p>RF-DETR finds the players; SAM2 carries them, prompted once on a frame
      chosen because all ten stand clear of each other. On the possession above,
      every prompted track survives the full 42.6 seconds — and identity comes
      from the shirts, not the tracker: jersey OCR voted over each track,
      constrained by the real roster.</p>
    </div>
    <div className="cell">
      <span className="mark">C</span>
      <h3>Camera motion, solved per frame</h3>
      <p>A keypoint model reads the court markings in every frame, so every frame gets its
      own homography and broadcast cuts stop mattering. Where the model fails, the
      fallback is one hand-calibrated frame carried by feature matching.</p>
    </div>
    <div className="cell">
      <span className="mark">T</span>
      <h3>Teams without labelling</h3>
      <p>Every torso crop becomes a SigLIP embedding; UMAP flattens them and
      K-means splits the two kits. Nobody types in a roster — on this clip the
      two sides separate unprompted, and the roster only enters afterwards, to
      turn numbers into names.</p>
    </div>
    <div className="cell">
      <span className="mark">S</span>
      <h3>Shot charts &amp; heat maps</h3>
      <p>Once every position is a floor coordinate, the shot chart is a scatter plot and
      the heat map is a histogram. The hard part was already done upstream.</p>
    </div>
    <div className="cell">
      <span className="mark">E</span>
      <h3>Nothing locked in</h3>
      <p>Per-frame positions as JSON, box score as CSV. It is your film and your numbers;
      take them to whatever you already use.</p>
    </div>
  </div>
</div></section>
    </>
  );
}
