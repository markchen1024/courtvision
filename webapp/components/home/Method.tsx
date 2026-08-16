export default function Method() {
  return (
    <>
<section className="band" id="method"><div className="shell">
  <div className="head-block">
    <span className="label">Method</span>
    <h2>Four steps, one of which is interesting.</h2>
    <p>Upload and detection are solved problems with good off-the-shelf parts. Step three
    is where the work is.</p>
  </div>

  <div className="steps">
    <div className="step">
      <span className="n">01</span>
      <h3>Upload the film</h3>
      <p>Any fixed camera that pans and zooms. Phone on a tripod at half court is the
      common case and it is fine.<code>ffmpeg → frames</code></p>
    </div>
    <div className="step">
      <span className="n">02</span>
      <h3>Detect and track</h3>
      <p>Every player boxed on every sampled frame, then linked into tracks that survive
      crossings and occlusion.<code>rf-detr → sam2 → tracks</code></p>
    </div>
    <div className="step">
      <span className="n">03</span>
      <h3>Register and calibrate</h3>
      <p>A keypoint model reads the court markings in every frame, so every frame
      gets its own homography and nobody clicks anything. Floors the model does
      not know fall back to one hand-calibrated reference carried by feature
      matching.<code>keypoints → H, per frame</code></p>
    </div>
    <div className="step">
      <span className="n">04</span>
      <h3>Read the floor</h3>
      <p>Each box&rsquo;s ground contact point goes through the transform and lands on the plan.
      From there it is ordinary basketball.<code>(u, v) → (x, y) in metres</code></p>
    </div>
  </div>
</div></section>
    </>
  );
}
