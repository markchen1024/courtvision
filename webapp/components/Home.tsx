'use client';

/* The marketing homepage, ported 1:1 from web/index.html. Markup and styles are
   transliterated, not redesigned, and the page logic runs exactly as it did --
   one imperative module against the same ids -- so the static original and this
   page cannot disagree while both exist. Componentisation can come later,
   section by section; fidelity came first. */

import { useEffect } from 'react';
import '@/app/theme.css';
import '@/app/plyr.css';
import '@/app/home.css';
import { initHome } from './homeScript';

export default function Home() {
  useEffect(() => initHome(), []);
  return (
    <>
<nav className="nav"><div className="shell bar">
  <a className="wordmark" href="/">
    <span className="name">court<b>vision</b></span>
    <span className="tag">AI</span>
  </a>
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
  <p className="fine">No account. Three minutes of the 2026 Summer League final, already processed.</p>

  {/* The product shot is the product. */}
  <div className="frame" id="product">
    <div className="chrome">
      <span className="dots"><i></i><i></i><i></i></span>
      <span className="url">courtvision.io/app — GSW @ MEM · 2026 Summer League final</span>
    </div>
    <div className="panes">
      <div className="pane">
        <div className="cap"><span className="label">Footage</span><span className="right" id="capClock">00:00</span></div>
        <div className="stage">
          <video id="film" muted loop playsInline autoPlay preload="metadata"></video>
          <div className="fallback" id="filmFallback" hidden>
            <div className="big">Clip not loaded</div>
            <div className="label">drop one at web/media/nba.mp4</div>
          </div>
        </div>
      </div>
      <div className="pane">
        <div className="cap"><span className="label">Court · metres</span><span className="right" id="capTracks">0 tracks</span></div>
        <canvas id="court"></canvas>
      </div>
    </div>

    <div className="tabs" role="tablist" aria-label="Statistics">
      <button role="tab" id="tab-box"    aria-controls="p-box"    aria-selected="true">Box score</button>
      <button role="tab" id="tab-team"   aria-controls="p-team"   aria-selected="false">Team stats</button>
      <button role="tab" id="tab-shots"  aria-controls="p-shots"  aria-selected="false">Shot chart</button>
      <button role="tab" id="tab-clips"  aria-controls="p-clips"  aria-selected="false">Timeline</button>
      <button role="tab" id="tab-mins"   aria-controls="p-mins"   aria-selected="false">Minutes &amp; impact</button>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-box" aria-labelledby="tab-box">
      <div className="clubs" role="tablist" aria-label="Club">
        <button role="tab" aria-selected="true" className="on gsw" data-club="0">Warriors — 94</button>
        <button role="tab" aria-selected="false" className="mem" data-club="1">Grizzlies — 90</button>
      </div>
      <div className="scroll-x"><table className="stat" id="boxTable"></table></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-team" aria-labelledby="tab-team" hidden>
      <div className="compare" id="compare"></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-shots" aria-labelledby="tab-shots" hidden>
      <div className="pillrow">
        <div className="clubs" role="tablist" aria-label="Club">
          <button role="tab" aria-selected="true" className="on gsw" data-shotclub="gsw">Warriors</button>
          <button role="tab" aria-selected="false" className="mem" data-shotclub="mem">Grizzlies</button>
        </div>
      </div>
      <div className="shots">
        <div>
          <canvas id="shotChart"></canvas>
          <div className="legend">
            <span><i className="made"></i>Made</span>
            <span><i className="miss"></i>Missed</span>
            <span id="shotTotal"></span>
          </div>
        </div>
        <div className="zones" id="zones"></div>
      </div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-clips" aria-labelledby="tab-clips" hidden>
      <div className="clips" id="clips"></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-mins" aria-labelledby="tab-mins" hidden>
      <div className="scroll-x"><table className="stat" id="minsTable"></table></div>
    </div>
  </div>

  <p className="tabnote">
    Positions and shot locations are measured by the pipeline. The box score is the
    official ESPN box for this game, standing in until tracked stats are labelled; the
    event labels are hand-tagged — see
    <a href="#limits">Limits</a>.
  </p>

  <div className="figures">
    <div><div className="n" id="figCourt">28 × 15</div><div className="k">metre court model</div></div>
    <div><div className="n" id="figHz">6 Hz</div><div className="k">positions sampled per second</div></div>
    <div><div className="n">0</div><div className="k">human clicks in the calibration</div></div>
    <div><div className="n">16</div><div className="k">court landmarks decoded for the solver</div></div>
  </div>
</div></header>

<main>

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
      <p>RF-DETR for players. Identity is associated <em>after</em> the homography, in
      court space, where the camera's motion is already divided out. Raced against
      ByteTrack on the same detections: 12.2 s median identity against 7.0 s.</p>
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
      <p>A vision-language model scores every shirt against colour words, and the pair
      that splits the players best decides the sides. Nobody types in a roster — on this
      clip it picked the two kits unprompted.</p>
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
      common case and it is fine.<code>ffmpeg → frames @ 6 Hz</code></p>
    </div>
    <div className="step">
      <span className="n">02</span>
      <h3>Detect and track</h3>
      <p>Every player boxed on every sampled frame, then linked into tracks that survive
      crossings and occlusion.<code>rf-detr → deep-eiou → tracks</code></p>
    </div>
    <div className="step">
      <span className="n">03</span>
      <h3>Register and calibrate</h3>
      <p>One reference frame is calibrated against the court plan. Every other frame is
      registered back to it with ORB/SIFT + RANSAC, and the two homographies
      compose.<code>H_court ∘ H_frame→ref</code></p>
    </div>
    <div className="step">
      <span className="n">04</span>
      <h3>Read the floor</h3>
      <p>Each box's ground contact point goes through the transform and lands on the plan.
      From there it is ordinary basketball.<code>(u, v) → (x, y) in metres</code></p>
    </div>
  </div>
</div></section>

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

<section className="closer"><div className="shell">
  <h2>Point it at your film.</h2>
  <p>The demo runs on a broadcast clip because that is where today's court models work.
  Community footage — phone camera, strange floor, no graphics team — is the harder half,
  and the repo measures exactly where it breaks.</p>
  <a className="btn primary lg" href="#product">Open the demo <span className="arrow">→</span></a>
</div></section>

</main>

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
