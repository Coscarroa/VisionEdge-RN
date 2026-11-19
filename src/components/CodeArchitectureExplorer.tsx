import React, { useState } from 'react';
import { Check, Copy, Download, FileCode2, FolderTree } from 'lucide-react';
import {
  DeliverableCategory,
  RN_ARCHITECTURE_FILES,
  RN_PROJECT_TREE,
} from '../data/rnArchitectureFiles';

interface CodeArchitectureExplorerProps {
  selectedFileId: string;
  onSelectFile: (fileId: string) => void;
  activeCategoryFilter: DeliverableCategory | 'all';
  onChangeCategoryFilter: (cat: DeliverableCategory | 'all') => void;
}

export const CodeArchitectureExplorer: React.FC<CodeArchitectureExplorerProps> = ({
  selectedFileId,
  onSelectFile,
  activeCategoryFilter,
  onChangeCategoryFilter,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredFiles =
    activeCategoryFilter === 'all'
      ? RN_ARCHITECTURE_FILES
      : RN_ARCHITECTURE_FILES.filter((f) => f.deliverable === activeCategoryFilter);

  const activeFile =
    RN_ARCHITECTURE_FILES.find((f) => f.id === selectedFileId) ??
    filteredFiles[0] ??
    RN_ARCHITECTURE_FILES[0];

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1800);
    } catch {
      // Clipboard fallback ignored if blocked
    }
  };

  const handleDownloadAll = () => {
    const sections = RN_ARCHITECTURE_FILES.map(
      (f) => `## ${f.path}\n\n${f.summary}\n\n\`\`\`${f.language}\n${f.content}\n\`\`\`\n`
    ).join('\n---\n\n');

    const bundle = `# VisionEdge RN — Arquitectura Completa On-Device Computer Vision\n\n## 1. Estructura del Proyecto\n\n\`\`\`text\n${RN_PROJECT_TREE}\n\`\`\`\n\n${sections}`;
    const blob = new Blob([bundle], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'VisionEdgeRN_Architecture.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  const categories: Array<{ id: DeliverableCategory | 'all'; label: string }> = [
    { id: 'all', label: 'Todos los Entregables' },
    { id: 'project', label: '1. Estructura & Deps' },
    { id: 'inference', label: '2. Inferencia & NMS' },
    { id: 'ui', label: '3. UI Skia 60 FPS' },
    { id: 'native', label: 'Swift & Kotlin' },
    { id: 'colab', label: 'Colab Export (.tflite)' },
  ];

  const codeLines = activeFile.content.split('\n');

  return (
    <div className="flex flex-col gap-6">
      {/* Header & Deliverable Filter Tabs */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            Entregables de Código Fuente React Native, Swift y Kotlin
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Módulos completos listos para compilación con VisionCamera v4, Fast TFLite (CoreML/NNAPI) y Skia.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap items-center gap-1 p-1 bg-[#121824] border border-white/10 rounded-xl">
            {categories.map((cat) => (
              <button
                key={cat.id}
                type="button"
                onClick={() => {
                  onChangeCategoryFilter(cat.id);
                  const firstMatch =
                    cat.id === 'all'
                      ? RN_ARCHITECTURE_FILES[0]
                      : RN_ARCHITECTURE_FILES.find((f) => f.deliverable === cat.id);
                  if (firstMatch) onSelectFile(firstMatch.id);
                }}
                className={`min-h-[36px] px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  activeCategoryFilter === cat.id
                    ? 'bg-sky-500 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={handleDownloadAll}
            className="min-h-[40px] px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-white/10 text-xs font-semibold text-slate-100 flex items-center gap-2 transition-colors cursor-pointer whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5 text-sky-400" />
            <span>Exportar Bundle (.md)</span>
          </button>
        </div>
      </div>

      {/* Main Split: Tree + File List on Left, Full Source Code on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Sidebar: Project Structure + Module Selector */}
        <div className="lg:col-span-4 flex flex-col gap-5">
          {/* File Selector List */}
          <div className="bg-[#121824] border border-white/10 rounded-2xl p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between pb-2 border-b border-white/10">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                <FileCode2 className="w-4 h-4 text-sky-400" />
                Módulos del Proyecto
              </span>
              <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                {filteredFiles.length} archivos
              </span>
            </div>

            <div className="flex flex-col gap-1">
              {filteredFiles.map((file) => {
                const isSelected = file.id === activeFile.id;
                return (
                  <button
                    key={file.id}
                    type="button"
                    onClick={() => onSelectFile(file.id)}
                    className={`w-full text-left px-3 py-2.5 rounded-xl transition-colors cursor-pointer flex flex-col gap-0.5 ${
                      isSelected
                        ? 'bg-sky-500/15 border border-sky-400/35 text-slate-100'
                        : 'border border-transparent hover:bg-white/[0.03] text-slate-300'
                    }`}
                  >
                    <span className="text-xs font-semibold font-mono truncate">{file.name}</span>
                    <span className="text-[11px] text-slate-400 truncate">{file.path}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Complete Folder Tree (Deliverable 1) */}
          <div className="bg-[#121824] border border-white/10 rounded-2xl p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                <FolderTree className="w-4 h-4 text-sky-400" />
                Árbol de Carpetas (rn-vision-edge)
              </span>
              <button
                type="button"
                onClick={() => handleCopy(RN_PROJECT_TREE, 'tree')}
                className="text-[11px] text-sky-400 hover:text-sky-300 flex items-center gap-1 cursor-pointer"
              >
                {copiedId === 'tree' ? (
                  <>
                    <Check className="w-3 h-3" /> Copiado
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" /> Copiar
                  </>
                )}
              </button>
            </div>
            <pre className="text-[11px] leading-relaxed font-mono text-slate-300 bg-slate-950/90 border border-white/5 rounded-xl p-3 overflow-x-auto">
              {RN_PROJECT_TREE}
            </pre>
          </div>
        </div>

        {/* Right Code Viewer */}
        <div className="lg:col-span-8 bg-[#121824] border border-white/10 rounded-2xl overflow-hidden flex flex-col">
          {/* File Header */}
          <div className="p-5 border-b border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex flex-col gap-1 min-w-0">
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                <span className="text-sky-400 font-semibold">{activeFile.path}</span>
                <span aria-hidden="true">·</span>
                <span>{activeFile.language}</span>
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">{activeFile.summary}</p>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pt-1 text-xs text-slate-400">
                {activeFile.highlights.map((h, i) => (
                  <span key={i}>• {h}</span>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => handleCopy(activeFile.content, activeFile.id)}
              className="min-h-[40px] px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold flex items-center gap-2 transition-colors cursor-pointer shrink-0 self-start"
            >
              {copiedId === activeFile.id ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Copiado</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copiar Código</span>
                </>
              )}
            </button>
          </div>

          {/* Line-Numbered Source Block */}
          <div className="bg-[#090D16] overflow-x-auto max-h-[680px] overflow-y-auto">
            <table className="w-full border-collapse font-mono text-[12px] leading-[1.6]">
              <tbody>
                {codeLines.map((line, index) => (
                  <tr key={index} className="hover:bg-white/[0.025]">
                    <td className="select-none text-right pr-4 pl-4 text-slate-600 w-12 align-top tabular-nums">
                      {index + 1}
                    </td>
                    <td className="pr-5 text-slate-200 whitespace-pre">{line || ' '}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
