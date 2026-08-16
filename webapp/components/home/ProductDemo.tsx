export default function ProductDemo() {
  return (
    <>
  {/* The product shot is the product. */}
  <div className="frame" id="product">
    <div className="chrome">
      <span className="dots"><i></i><i></i><i></i></span>
      <span className="url">courtvision.io/app — NYK @ DET · 2025 East first round, game 4</span>
    </div>
    <div className="panes">
      <div className="pane">
        <div className="cap"><span className="label">Footage</span>
          <span className="viewtoggle" role="tablist" aria-label="Footage view">
            <button role="tab" aria-selected="true" className="on" data-view="ai">AI view</button>
            <button role="tab" aria-selected="false" data-view="raw">Broadcast</button>
          </span>
          <span className="right" id="capClock">00:00</span></div>
        <div className="stage">
          <video id="film" muted loop playsInline autoPlay preload="metadata" poster="/media/poster.jpg"></video>
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
        <button role="tab" aria-selected="true" className="on nyk" data-club="0">Knicks — 94</button>
        <button role="tab" aria-selected="false" className="det" data-club="1">Pistons — 93</button>
      </div>
      <div className="scroll-x"><table className="stat" id="boxTable"></table></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-team" aria-labelledby="tab-team" hidden>
      <div className="compare" id="compare"></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-shots" aria-labelledby="tab-shots" hidden>
      <div className="pillrow">
        <div className="clubs" role="tablist" aria-label="Club">
          <button role="tab" aria-selected="true" className="on nyk" data-shotclub="nyk">Knicks</button>
          <button role="tab" aria-selected="false" className="det" data-shotclub="det">Pistons</button>
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
      <p className="fine">The official ESPN play-by-play for this game, all four
        quarters. Rows marked <em>on film</em> fall inside the possession above,
        aligned by the broadcast scoreboard clock — nothing here is read from
        the footage; see <a href="#limits">Limits</a>.</p>
      <div className="clips" id="clips"></div>
    </div>

    <div className="tabpanel" role="tabpanel" id="p-mins" aria-labelledby="tab-mins" hidden>
      <div className="clubs" role="tablist" aria-label="Club">
        <button role="tab" aria-selected="true" className="on nyk" data-minsclub="0">Knicks</button>
        <button role="tab" aria-selected="false" className="det" data-minsclub="1">Pistons</button>
      </div>
      <div className="scroll-x"><table className="stat" id="minsTable"></table></div>
    </div>
  </div>

  <p className="tabnote">
    Positions are measured by the pipeline, and the names in the AI view are read
    off the jerseys — number OCR plus team clustering, with unresolved tracks left
    anonymous rather than guessed; on this possession that measures 100% precision
    and 98.8% coverage against hand ground truth. The box score and timeline are
    the official ESPN record for this game; the shot chart is illustrative — see{' '}
    <a href="#limits">Limits</a>.
  </p>

  <div className="figures">
    <div><div className="n" id="figCourt">28 × 15</div><div className="k">metre court model</div></div>
    <div><div className="n" id="figHz">5 Hz</div><div className="k">positions sampled per second</div></div>
    <div><div className="n">0</div><div className="k">human clicks in the calibration</div></div>
    <div><div className="n" id="figSolved">1277/1277</div><div className="k">frames court-solved on this clip</div></div>
  </div>
    </>
  );
}
