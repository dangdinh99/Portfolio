import type { ReactNode } from 'react';
import { useRevealOnScroll } from '../hooks/useRevealOnScroll';

type Props = { className?: string; children: ReactNode };

export function SectionReveal({ className = '', children }: Props) {
  const { ref, visible } = useRevealOnScroll<HTMLDivElement>();
  return (
    <div ref={ref} className={`${className} ${visible ? 'fade-in-up' : ''}`.trim()}>
      {children}
    </div>
  );
}
