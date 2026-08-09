// The write-back half of the review UI: a human's identity decision lands in
// web/data/nba.json (the same file the viewer and homepage read -- fs.writeFile
// truncates in place, so the hardlink into public/data stays on the same
// inode), the review manifest is kept in step, and every decision is appended
// to an audit file so hand corrections are never silently mixed into model
// output.
import { promises as fs } from 'fs';
import path from 'path';

const ROOT = path.join(process.cwd(), '..', 'web', 'data');
const DATA = path.join(ROOT, 'nba.json');
const MANIFEST = path.join(ROOT, 'review', 'manifest.json');
const AUDIT = path.join(ROOT, 'corrections_nba.json');

type Body = {
  trackId: number;
  action: 'assign' | 'ignore' | 'unassign';
  number?: string;
  name?: string;
  team?: 'home' | 'away';
  // for unassign: what the track falls back to (its OCR number, if any)
  ocrNumber?: string | null;
};

export async function POST(req: Request) {
  const body = (await req.json()) as Body;
  if (typeof body.trackId !== 'number' ||
      !['assign', 'ignore', 'unassign'].includes(body.action)) {
    return Response.json({ error: 'bad request' }, { status: 400 });
  }
  if (body.action === 'assign' && !body.number) {
    return Response.json({ error: 'assign needs a number' }, { status: 400 });
  }

  const doc = JSON.parse(await fs.readFile(DATA, 'utf8'));
  const player = doc.players.find((p: { id: number }) => p.id === body.trackId);
  if (!player) {
    return Response.json({ error: 'unknown track' }, { status: 404 });
  }

  if (body.action === 'assign') {
    player.number = String(body.number);
    if (body.name) player.name = body.name;
    else delete player.name;
    if (body.team) player.team = body.team;
    player.identity = 'human';
  } else if (body.action === 'unassign') {
    // conflict resolution strips a wrong name; the track rejoins the queue
    delete player.name;
    player.number = body.ocrNumber ? String(body.ocrNumber) : `T${player.id}`;
    delete player.identity;
  } else {
    player.identity = 'ignored';
  }
  await fs.writeFile(DATA, JSON.stringify(doc));

  try {
    const manifest = JSON.parse(await fs.readFile(MANIFEST, 'utf8'));
    const track = manifest.tracks.find(
      (t: { id: number }) => t.id === body.trackId);
    if (track) {
      track.status = body.action === 'ignore' ? 'ignored'
        : body.action === 'unassign'
          ? (track.ocr?.number ? 'number-only' : 'anonymous')
          : 'human';
      track.number = player.number;
      track.name = player.name ?? null;
      track.team = player.team;
      await fs.writeFile(MANIFEST, JSON.stringify(manifest));
    }
  } catch {
    // a missing manifest only degrades the queue view, never the data write
  }

  let audit: unknown[] = [];
  try {
    audit = JSON.parse(await fs.readFile(AUDIT, 'utf8'));
  } catch {}
  audit.push({ ...body, at: new Date().toISOString() });
  await fs.writeFile(AUDIT, JSON.stringify(audit, null, 1));

  return Response.json({ ok: true, player });
}
