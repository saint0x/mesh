import { useEffect, useRef, useState } from 'react'
import { ArchitectureVisual } from '../architecture/ArchitectureVisuals'
import { architectureSlides } from '../architecture/slides'
import type { DashboardPageProps } from '../lib/pageProps'

const swipeThreshold = 42

export function ArchitecturePage({ controller }: DashboardPageProps) {
  const [activeIndex, setActiveIndex] = useState(0)
  const pointerStartX = useRef<number | null>(null)
  const pointerId = useRef<number | null>(null)
  const selectedNetwork = controller.selectedNetwork
  const localDevice = controller.networkDevices.find((device) => device.localDevice)
  const selectedTopology = controller.selectedTopology
  const initialSlide = architectureSlides[0] ?? null

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT')
      ) {
        return
      }
      if (event.key === 'ArrowRight') {
        setActiveIndex((current) => Math.min(current + 1, architectureSlides.length - 1))
      }
      if (event.key === 'ArrowLeft') {
        setActiveIndex((current) => Math.max(current - 1, 0))
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  if (!initialSlide) {
    return null
  }

  const activeSlide = architectureSlides[activeIndex] ?? initialSlide

  return (
    <div className="dashboard-stack architecture-stack">
      <section className="panel dashboard-panel architecture-hero">
        <div className="architecture-hero-copy">
          <div className="eyebrow">System narrative</div>
          <h2>How one request moves through MeshNet and the ZIP engine</h2>
          <p className="dashboard-panel-copy">
            This walkthrough starts at the public MeshNet surface, crosses the durable control plane,
            enters the agent-embedded ZIP runtime, and returns through the ledger and dashboard.
          </p>
        </div>
        <div className="architecture-hero-stats">
          <article>
            <span>Live network</span>
            <strong>{selectedNetwork?.name ?? 'No network selected'}</strong>
            <small>{selectedNetwork?.preferredPath ?? 'No preferred path available'}</small>
          </article>
          <article>
            <span>Healthy devices</span>
            <strong>{controller.summary.healthyDevices}</strong>
            <small>{controller.networkDevices.length} devices in current mesh scope</small>
          </article>
          <article>
            <span>Topology posture</span>
            <strong>{selectedTopology?.ringStable ? 'stable ring' : 'pending ring'}</strong>
            <small>{selectedTopology?.workers.length ?? 0} workers in active topology snapshot</small>
          </article>
          <article>
            <span>Local runtime</span>
            <strong>{localDevice?.capabilities.defaultExecutionProvider ?? 'n/a'}</strong>
            <small>{localDevice?.name ?? 'local device not present in this network'}</small>
          </article>
        </div>
      </section>

      <section className="panel dashboard-panel architecture-carousel-panel">
        <div className="architecture-carousel-head">
          <div>
            <div className="eyebrow">{activeSlide.eyebrow}</div>
            <h3>{activeSlide.title}</h3>
          </div>
          <div className="architecture-carousel-meta">
            <span>Swipe, click, or use arrow keys</span>
            <strong>
              {activeIndex + 1} / {architectureSlides.length}
            </strong>
          </div>
        </div>

        <div
          className="architecture-carousel"
          onPointerDown={(event) => {
            pointerStartX.current = event.clientX
            pointerId.current = event.pointerId
            event.currentTarget.setPointerCapture(event.pointerId)
          }}
          onPointerUp={(event) => {
            if (pointerId.current !== event.pointerId || pointerStartX.current === null) {
              return
            }
            const delta = event.clientX - pointerStartX.current
            if (delta <= -swipeThreshold) {
              setActiveIndex((current) => Math.min(current + 1, architectureSlides.length - 1))
            } else if (delta >= swipeThreshold) {
              setActiveIndex((current) => Math.max(current - 1, 0))
            }
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
            pointerStartX.current = null
            pointerId.current = null
          }}
          onPointerCancel={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
            pointerStartX.current = null
            pointerId.current = null
          }}
        >
          <div
            className="architecture-carousel-track"
            style={{ transform: `translateX(-${activeIndex * 100}%)` }}
          >
            {architectureSlides.map((slide, index) => (
              <article key={slide.id} className="architecture-slide">
                <div className="architecture-slide-grid">
                  <div className="architecture-slide-copy">
                    <p className="architecture-slide-summary">{slide.summary}</p>

                    <div className="architecture-pillars">
                      <section>
                        <span>MeshNet platform layer</span>
                        <p>{slide.platformDetail}</p>
                      </section>
                      <section>
                        <span>ZIP engine layer</span>
                        <p>{slide.zipDetail}</p>
                      </section>
                    </div>

                    <section className="architecture-operator-card">
                      <span>Why this matters to an operator</span>
                      <p>{slide.operatorLens}</p>
                    </section>

                    <section className="architecture-code-card">
                      <span>Primary code anchors</span>
                      <div className="architecture-code-list">
                        {slide.codeRefs.map((codeRef) => (
                          <code key={codeRef}>{codeRef}</code>
                        ))}
                      </div>
                    </section>
                  </div>

                  <div className="architecture-slide-visual">
                    <ArchitectureVisual slide={slide} index={index} />
                  </div>
                </div>

                <div className="architecture-metric-strip">
                  {slide.metrics.map((metric) => (
                    <article key={metric.label}>
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                    </article>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className="architecture-carousel-controls">
          <button
            className="ghost-button small"
            onClick={() => setActiveIndex((current) => Math.max(current - 1, 0))}
            disabled={activeIndex === 0}
          >
            Previous
          </button>
          <div className="architecture-step-rail">
            {architectureSlides.map((slide, index) => (
              <button
                key={slide.id}
                className={index === activeIndex ? 'architecture-step active' : 'architecture-step'}
                onClick={() => setActiveIndex(index)}
                aria-label={`Open ${slide.title}`}
                aria-current={index === activeIndex ? 'step' : undefined}
              >
                <strong>{index + 1}</strong>
                <span>{slide.title}</span>
              </button>
            ))}
          </div>
          <button
            className="primary-button"
            onClick={() =>
              setActiveIndex((current) => Math.min(current + 1, architectureSlides.length - 1))
            }
            disabled={activeIndex === architectureSlides.length - 1}
          >
            Next step
          </button>
        </div>
      </section>
    </div>
  )
}
