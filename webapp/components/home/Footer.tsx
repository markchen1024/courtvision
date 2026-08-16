export default function Footer() {
  return (
    <>
<footer><div className="shell">
  <div className="foot">
    <div className="about">
      <span className="wordmark">
        <span className="name">court<b>vision</b></span>
        <span className="tag">AI</span>
      </span>
      <p>Turning basketball footage into a live stat line, one homography at a time.</p>
    </div>
    <div>
      <h4>Product</h4>
      <ul>
        <li><a href="#product">Open the demo</a></li>
        <li><a href="#method">Method</a></li>
        <li><a href="#limits">Limits</a></li>
        <li><a href="#pricing">Pricing</a></li>
      </ul>
    </div>
    <div>
      <h4>Developers</h4>
      <ul>
        <li><a href="#">Docs</a></li>
        <li><a href="#">JSON schema</a></li>
        <li><a href="#">Changelog</a></li>
        <li><a href="#">Status</a></li>
      </ul>
    </div>
    <div>
      <h4>Company</h4>
      <ul>
        <li><a href="#">About</a></li>
        <li><a href="#">Contact</a></li>
        <li><a href="#">Privacy</a></li>
        <li><a href="#">Terms</a></li>
      </ul>
    </div>
  </div>
  <div className="colophon">
    <span>A portfolio demo — every link but the demo is a placeholder, and nothing is for sale.</span>
    <span className="spacer"></span>
    <span id="colophonData">…</span>
  </div>
</div></footer>
    </>
  );
}
