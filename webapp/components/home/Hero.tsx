import ProductDemo from './ProductDemo';

export default function Hero() {
  return (
    <>
<header className="hero"><div className="shell">
  <span className="eyebrow"><i></i>Real footage · real geometry</span>
  <h1>Your game film already <em>has</em> the stats in it.</h1>
  <p className="sub">
    courtvision detects every player on the floor and maps the camera onto a top-down
    court plan, so a body at pixel (1284, 613) becomes a player standing 6.4 metres from
    the basket. Spacing, shot charts and heat maps all fall out of having real coordinates.
  </p>
  <div className="cta">
    <a className="btn primary lg" href="#product">Open the demo <span className="arrow">→</span></a>
    <a className="btn lg" href="#method">See how it works</a>
  </div>
  <p className="fine">No account. A full 42.6-second possession of NYK @ DET, game 4 of the 2025 East first round, already processed — every label measured at 100% precision against hand ground truth.</p>
  <ProductDemo />
</div></header>
    </>
  );
}
