import type { Metadata } from 'next';
import Review from '@/components/Review';
import '../theme.css';
import './review.css';

export const metadata: Metadata = {
  title: 'courtvision — identity review',
  description:
    'The human half of player identification: AI resolves what it can, ' +
    'the rest is cleared here and marked as hand-corrected.',
};

export default function ReviewPage() {
  return <Review />;
}
