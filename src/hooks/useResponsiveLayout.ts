import { useEffect, useMemo, useState } from 'react';

export interface SafeAreaInsets {
  top: number;
  bottom: number;
}

export interface ResponsiveLayoutMetrics {
  width: number;
  height: number;
  insets: SafeAreaInsets;
  usableHeight: number;
  isCompactScreen: boolean;
  gap: number;
  paddingHorizontal: number;
  topBarHeight: number;
  counterRowHeight: number;
  counterFontSize: number;
  metricsRowHeight: number;
  waveHeight: number;
  heatmapHeight: number;
  trackingChartHeight: number;
  trackingSvgHeight: number;
  footerHeight: number;
  estimatedViewportHeight: number;
}

function readCssSafeAreaInsets(): SafeAreaInsets {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return { top: 0, bottom: 0 };
  }
  const probe = document.createElement('div');
  probe.style.position = 'fixed';
  probe.style.top = '0px';
  probe.style.left = '0px';
  probe.style.paddingTop = 'env(safe-area-inset-top, 0px)';
  probe.style.paddingBottom = 'env(safe-area-inset-bottom, 0px)';
  probe.style.visibility = 'hidden';
  probe.style.pointerEvents = 'none';
  document.body.appendChild(probe);
  const computed = window.getComputedStyle(probe);
  const top = parseFloat(computed.paddingTop) || 0;
  const bottom = parseFloat(computed.paddingBottom) || 0;
  document.body.removeChild(probe);
  return { top, bottom };
}

export function useResponsiveLayout(): ResponsiveLayoutMetrics {
  const [dimensions, setDimensions] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 390,
    height: typeof window !== 'undefined' ? window.innerHeight : 844,
    insets: readCssSafeAreaInsets(),
  }));

  useEffect(() => {
    const sync = () => {
      const nextHeight = window.visualViewport?.height ?? window.innerHeight;
      const nextWidth = window.visualViewport?.width ?? window.innerWidth;
      setDimensions({
        width: Math.round(nextWidth),
        height: Math.round(nextHeight),
        insets: readCssSafeAreaInsets(),
      });
    };

    sync();
    window.addEventListener('resize', sync);
    window.visualViewport?.addEventListener('resize', sync);
    return () => {
      window.removeEventListener('resize', sync);
      window.visualViewport?.removeEventListener('resize', sync);
    };
  }, []);

  return useMemo(() => {
    const { width, height, insets } = dimensions;
    const usableHeight = Math.max(320, height - insets.top - insets.bottom);
    const isCompactScreen = height < 720;

    const gap = isCompactScreen ? 6 : 10;
    const paddingHorizontal = 12;
    const topBarHeight = isCompactScreen ? 36 : 42;
    const counterRowHeight = isCompactScreen ? 64 : 76;
    const counterFontSize = isCompactScreen ? 20 : 26;
    const metricsRowHeight = isCompactScreen ? 82 : 96;
    const waveHeight = isCompactScreen ? 24 : 34;
    const heatmapHeight = isCompactScreen ? 28 : 38;
    const trackingChartHeight = isCompactScreen ? 88 : 105;
    const trackingSvgHeight = isCompactScreen ? 38 : 50;
    const footerHeight = isCompactScreen ? 42 : 48;

    // 6 bandas -> 5 huecos verticales de tamaño `gap`
    const fixedHeightSum =
      topBarHeight +
      counterRowHeight +
      metricsRowHeight +
      trackingChartHeight +
      footerHeight +
      gap * 5;

    const estimatedViewportHeight = Math.max(120, usableHeight - fixedHeightSum);

    return {
      width,
      height,
      insets,
      usableHeight,
      isCompactScreen,
      gap,
      paddingHorizontal,
      topBarHeight,
      counterRowHeight,
      counterFontSize,
      metricsRowHeight,
      waveHeight,
      heatmapHeight,
      trackingChartHeight,
      trackingSvgHeight,
      footerHeight,
      estimatedViewportHeight,
    };
  }, [dimensions]);
}
