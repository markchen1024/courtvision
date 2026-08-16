import Link from 'next/link';

export default function Nav() {
  return (
    <>
<nav className="nav"><div className="shell bar">
  <Link className="wordmark" href="/">
    <span className="name">court<b>vision</b></span>
    <span className="tag">AI</span>
  </Link>
  <div className="links">
    <a href="#product">Product</a>
    <a href="#method">Method</a>
    <a href="#limits">Limits</a>
    <a href="#pricing">Pricing</a>
    <a href="#">Docs</a>
  </div>
  <div className="actions">
    <a className="btn ghost" href="#">Sign in</a>
    <a className="btn primary" href="#product">Open the demo <span className="arrow">→</span></a>
  </div>
</div></nav>
    </>
  );
}
