'use client';

import React from 'react';

function parseInlineMarkdown(text: string): React.ReactNode[] {
  const elements: React.ReactNode[] = [];
  const tokenRegex =
    /(`([^`]+)`)|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      elements.push(text.substring(lastIndex, match.index));
    }
    const key = `inline-${match.index}`;
    if (match[1]) {
      // `code`
      elements.push(
        <code key={key} className="bg-[#FFF6D8] text-[#B37D08] px-1.5 py-0.5 rounded text-xs font-mono">
          {match[2]}
        </code>
      );
    } else if (match[3]) {
      // **bold**
      elements.push(
        <strong key={key} className="font-bold text-[#101828]">
          {match[4]}
        </strong>
      );
    } else if (match[5]) {
      // *italic*
      elements.push(
        <em key={key} className="italic text-[#475467]">
          {match[6]}
        </em>
      );
    } else if (match[7]) {
      // [text](url)
      const linkText = match[8];
      const linkUrl = match[9];
      elements.push(
        <a
          key={key}
          href={linkUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[#D99E10] hover:text-[#B37D08] underline font-medium inline-flex items-center gap-0.5"
        >
          {linkText}
        </a>
      );
    }
    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    elements.push(text.substring(lastIndex));
  }

  return elements.length > 0 ? elements : [text];
}

export function MarkdownRenderer({ content }: { content: string }) {
  if (!content) return null;

  const rawLines = content.split('\n');
  const renderedNodes: React.ReactNode[] = [];

  let currentListItems: React.ReactNode[] = [];
  let isCurrentListOrdered = false;

  const flushList = (keyPrefix: string) => {
    if (currentListItems.length > 0) {
      if (isCurrentListOrdered) {
        renderedNodes.push(
          <ol key={`${keyPrefix}-ol`} className="list-decimal pl-5 my-1.5 space-y-1 text-[#344054]">
            {currentListItems}
          </ol>
        );
      } else {
        renderedNodes.push(
          <ul key={`${keyPrefix}-ul`} className="list-disc pl-5 my-1.5 space-y-1 text-[#344054]">
            {currentListItems}
          </ul>
        );
      }
      currentListItems = [];
    }
  };

  rawLines.forEach((line, lineIdx) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList(`flush-${lineIdx}`);
      return;
    }

    if (trimmed.startsWith('### ')) {
      flushList(`flush-${lineIdx}`);
      renderedNodes.push(
        <h4 key={`h3-${lineIdx}`} className="text-sm font-bold text-[#101828] mt-2 mb-0.5">
          {parseInlineMarkdown(trimmed.slice(4))}
        </h4>
      );
      return;
    }
    if (trimmed.startsWith('## ')) {
      flushList(`flush-${lineIdx}`);
      renderedNodes.push(
        <h3 key={`h2-${lineIdx}`} className="text-base font-bold text-[#101828] mt-3 mb-1">
          {parseInlineMarkdown(trimmed.slice(3))}
        </h3>
      );
      return;
    }
    if (trimmed.startsWith('# ')) {
      flushList(`flush-${lineIdx}`);
      renderedNodes.push(
        <h2 key={`h1-${lineIdx}`} className="text-lg font-extrabold text-[#101828] mt-3 mb-1">
          {parseInlineMarkdown(trimmed.slice(2))}
        </h2>
      );
      return;
    }

    const ulMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
    if (ulMatch) {
      if (isCurrentListOrdered && currentListItems.length > 0) {
        flushList(`switch-${lineIdx}`);
      }
      isCurrentListOrdered = false;
      currentListItems.push(
        <li key={`li-${lineIdx}`} className="leading-relaxed">
          {parseInlineMarkdown(ulMatch[2])}
        </li>
      );
      return;
    }

    const olMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
    if (olMatch) {
      if (!isCurrentListOrdered && currentListItems.length > 0) {
        flushList(`switch-${lineIdx}`);
      }
      isCurrentListOrdered = true;
      currentListItems.push(
        <li key={`oli-${lineIdx}`} className="leading-relaxed">
          {parseInlineMarkdown(olMatch[2])}
        </li>
      );
      return;
    }

    flushList(`flush-${lineIdx}`);
    renderedNodes.push(
      <p key={`p-${lineIdx}`} className="my-1 leading-relaxed text-[#344054]">
        {parseInlineMarkdown(line)}
      </p>
    );
  });

  flushList('final');

  return <div className="space-y-0.5">{renderedNodes}</div>;
}
