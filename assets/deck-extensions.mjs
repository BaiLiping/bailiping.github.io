// Companion notes live beneath their slide deck in the site's navigation.
// Shared by the static installer and decks with an authoring/build pipeline.
const home = { label: 'All topics', href: '/' };
export const deckExtensions = {
  'slam-slides': {
    label: 'SLAM',
    resources: [
      { title: 'Visual SLAM', href: '/vslam/', description: 'Step through tracking, loop closure, pose-graph optimization, and bundle adjustment.' },
      { title: 'Graph SLAM', href: '/graph-slam/', description: 'Worked numerical notebooks on graph optimization, iSAM, bundle adjustment, and structure from motion.' }
    ]
  },
  'frame-registration-slides': {
    label: 'Frame Registration',
    resources: [{ title: 'Extended notes & interactive examples', href: '/frame-registration/', description: 'Explore the objectives, assumptions, worked examples, and registration experiments in more detail.' }]
  },
  'target-handover-slides': {
    label: 'Target Handover', parent: home,
    resources: [{ title: 'Paper, code & tracking results', href: '/handover/', description: 'Read the project page and explore the paper, implementation, and simulation results.' }]
  },
  'bp-vs-pmbm-slides': {
    label: 'BP vs PMBM',
    resources: [{ title: 'Extended notes & association experiments', href: '/bp-vs-pmbm/', description: 'Compare association marginals and joint hypotheses through the full interactive explanation.' }]
  },
  'eo-mtt-slides': {
    label: 'Extended-Target Tracking', parent: home,
    resources: [{ title: 'Extended notes & partition experiments', href: '/eo-mtt/', description: 'Explore partition uncertainty, candidate generation, and inference methods in the extended notes.' }]
  },
  'multidensity-fusion': { label: 'Density Fusion', parent: home, resources: [] }
};

const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const anchor = (href, label) => `<a class="deck-extension-link" href="${escape(href)}">${escape(label)}</a>`;

export function applyDeckExtensions(doc, path) {
  const config = deckExtensions[path];
  if (!config) return doc;
  const { background = '#ffffff', color = '#16273e', accent = '#087f68' } = doc.theme;
  const sans = 'Arial, Helvetica, sans-serif';
  const titleElement = doc.slides.flatMap(s => s.elements).find(e =>
    e.type === 'text' && ['heading', 'htitle', 'slide-title', 'rungtitle'].includes(e.id));
  const headingFont = titleElement?.fontFamily || doc.theme.fontFamily || sans;
  const text = (id, x, y, w, h, html, size = 22, options = {}) => ({
    id, type: 'text', x, y, w, h, html, fontSize: size, fontFamily: sans,
    fontWeight: 400, color, align: 'left', valign: 'top', lineHeight: 1.3,
    rotation: 0, opacity: 1, ...options
  });
  const rect = (id, x, y, w, h, fill, stroke = 'none') => ({
    id, type: 'shape', shape: 'rect', x, y, w, h, fill, stroke,
    strokeWidth: stroke === 'none' ? 0 : 1, radius: 8, rotation: 0, opacity: 1
  });
  // Idempotent: appendices and shortcuts can be regenerated after any deck edit.
  doc.slides = doc.slides.filter(s => s.id !== 's-extensions');
  for (const slide of doc.slides) {
    slide.elements = slide.elements.filter(e => !e.id.startsWith('deck-extension-'));
    if (config.parent) slide.elements.push(text('deck-extension-parent', 72, 7, 710, 19,
      anchor(config.parent.href, '← ' + config.parent.label), 11, { color: accent }));
    if (config.resources.length) slide.elements.push(text('deck-extension-shortcut', 1060, 7, 148, 19,
      'EXTENSIONS →', 11, { color: accent, align: 'right', link: 's-extensions' }));
  }
  if (!config.resources.length) return doc;
  doc.slides.push({
    id: 's-extensions', background, transition: 'none',
    notes: `Extensions to the ${config.label} slides. Companion pages: ${config.resources.map(r => r.href).join(', ')}.`,
    elements: [
      text('deck-extension-kicker', 72, 35, 1136, 22, config.label.toUpperCase() + ' / EXTENSIONS', 12, { color: accent, fontWeight: 700 }),
      text('deck-extension-title', 72, 80, 1136, 70, 'Continue beyond the slides', 40, { fontFamily: headingFont, fontWeight: 700 }),
      rect('deck-extension-rule', 72, 165, 1136, 1, '#d8dedf'),
      text('deck-extension-intro', 72, 192, 1136, 42, 'Extended explanations, worked examples, and supporting material.', 21, { color: '#596d70' }),
      ...config.resources.flatMap((resource, i) => {
        const y = 268 + i * 154;
        return [
          rect('deck-extension-panel-' + i, 72, y, 1136, 132, '#ffffff', '#d8dedf'),
          text('deck-extension-link-' + i, 96, y + 22, 1088, 38, anchor(resource.href, resource.title + ' →'), 25, { color: accent, fontWeight: 700 }),
          text('deck-extension-description-' + i, 96, y + 73, 1056, 50, resource.description, 19, { color: '#596d70' })
        ];
      }),
      text('deck-extension-return', 72, 624, 1136, 34,
        anchor(config.parent?.href || '/', '← ' + (config.parent?.label || 'All topics')), 18, { color: accent }),
      text('deck-extension-footer', 72, 683, 960, 20, config.label + ' · Bai Liping', 11, { color: '#596d70' }),
      text('deck-extension-page', 1090, 683, 118, 20, '{{page}} / {{pages}}', 11, { color: '#596d70', align: 'right' })
    ]
  });
  return doc;
}

export function installExtensionAssets(html) {
  if (html.includes('src="../assets/deck-extensions.js"')) return html;
  return html.replace('</head>', '  <script type="module" src="../assets/deck-extensions.js"></script>\n</head>');
}
