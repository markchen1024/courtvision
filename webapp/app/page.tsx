import type { Metadata } from 'next';
import Home from '@/components/Home';

export const metadata: Metadata = {
  title: 'courtvision — game film into court coordinates',
  description:
    'Upload a game. courtvision detects every player, maps the camera onto a ' +
    'top-down court plan, and turns pixels into positions in metres.',
};

export default function Page() {
  return <Home />;
}
