import type { Metadata } from 'next';
import Viewer from '@/components/Viewer';

export const metadata: Metadata = {
  title: 'courtvision — viewer',
  description:
    'Game footage beside the measured top-down court, moving in step. ' +
    'Positions from the pipeline; events tagged by hand.',
};

export default function ViewerPage() {
  return <Viewer />;
}
