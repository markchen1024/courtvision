export default function Pricing() {
  return (
    <>
<section className="band" id="pricing"><div className="shell">
  <div className="head-block">
    <span className="label">Pricing</span>
    <h2>Priced per program, not per minute.</h2>
    <p>Film sessions should not come with a meter running. Illustrative pricing — this is a
    portfolio build and nothing here takes payment.</p>
  </div>

  <div className="plans">
    <div className="plan">
      <span className="tier">Film room</span>
      <div className="price">$0</div>
      <p className="blurb">One team, one season, enough to find out whether the geometry holds
      up on your court.</p>
      <ul>
        <li>3 uploads a month</li>
        <li>Top-down view and player traces</li>
        <li>CSV export</li>
        <li>Community support</li>
      </ul>
      <a className="btn" href="#">Start free</a>
    </div>

    <div className="plan featured">
      <span className="tier">Program</span>
      <div className="price">$79 <span>/ month</span></div>
      <p className="blurb">For a club running weekly film. Unlimited uploads and everything the
      coordinates unlock.</p>
      <ul>
        <li>Unlimited uploads, 1080p</li>
        <li>Shot charts and heat maps</li>
        <li>Automatic team assignment</li>
        <li>JSON API and webhooks</li>
        <li>Season aggregation</li>
      </ul>
      <a className="btn primary" href="#">Start 14-day trial</a>
    </div>

    <div className="plan">
      <span className="tier">League</span>
      <div className="price">Custom</div>
      <p className="blurb">Many venues, many cameras, one set of numbers everybody trusts.</p>
      <ul>
        <li>Multi-team and multi-venue</li>
        <li>On-site camera calibration</li>
        <li>SSO and audit log</li>
        <li>Data residency options</li>
      </ul>
      <a className="btn" href="#">Talk to us</a>
    </div>
  </div>
</div></section>
    </>
  );
}
