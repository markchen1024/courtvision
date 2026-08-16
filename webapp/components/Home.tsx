'use client';

/* The marketing homepage. The markup was ported 1:1 from web/index.html and
   has now been sectioned into components/home/* -- pure moves, every id and
   class unchanged -- while the page logic stays one imperative module
   (homeScript.js) driving those same ids. That split is deliberate: the
   script predates React here, it is tested against the static original, and
   rewriting working DOM logic into hooks is a refactor this demo does not
   need. The components give the file layout a table of contents; the script
   remains the single driver. */

import { useEffect } from 'react';
import '@/app/theme.css';
import '@/app/plyr.css';
import '@/app/home.css';
import { initHome } from './homeScript';

import Nav from './home/Nav';
import Hero from './home/Hero';
import WhatItDoes from './home/WhatItDoes';
import Method from './home/Method';
import Limits from './home/Limits';
import Pricing from './home/Pricing';
import Closer from './home/Closer';
import Footer from './home/Footer';

export default function Home() {
  useEffect(() => initHome(), []);
  return (
    <>
      <Nav />
      <Hero />
      <main>
        <WhatItDoes />
        <Method />
        <Limits />
        <Pricing />
        <Closer />
      </main>
      <Footer />
    </>
  );
}
