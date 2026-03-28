import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rawPath = path.join(__dirname, '../src/data/_projectsRaw.txt');
const outPath = path.join(__dirname, '../src/data/projects.ts');

let raw = fs.readFileSync(rawPath, 'utf8');
raw = raw.replace(/'image\/campaignzero.png '/g, "'image/campaignzero.png'");

const project6 = `,
  'project6': {
    title: 'PC Butler: Local PC Health Agent',
    category: 'Systems | Automation',
    image: 'image/treeornot.jpg',
    description: 'Python agent that collects telemetry, scores system health, and runs cataloged maintenance actions with a local SQLite dashboard.',
    longDescription: '<p><strong>Overview:</strong> A personal desktop companion that monitors disk, network, and performance signals, persists history in SQLite, and surfaces a local Streamlit dashboard for health trends and orchestrated fixes.</p><p><strong>Stack:</strong> Python telemetry collectors, SQLite, and Streamlit UI—designed to run locally without shipping data to the cloud by default.</p>',
    tech: ['Python', 'SQLite', 'Streamlit', 'psutil'],
    features: [
      'Telemetry pipeline for machine signals',
      'Health scoring and history in SQLite',
      'Dashboard for at-a-glance status',
      'Action library for cleanup and maintenance tasks',
    ],
    stats: [
      { value: 'Local', label: 'First' },
      { value: 'SQLite', label: 'Storage' },
    ],
    github: 'https://github.com/dangdinh99/pcbutler',
    demo: null,
    report: null,
    hasGithub: true,
    hasDemo: false,
    hasReport: false,
  }`;

const lastBrace = raw.lastIndexOf('\r\n}');
if (lastBrace === -1) throw new Error('no closing brace');
const inner = raw.slice(0, lastBrace) + project6 + raw.slice(lastBrace);

const header = `import type { Project } from './projectTypes';

export const projectsData: Record<string, Omit<Project, 'id'>> = 
`;

const footer = `;

export const projects: Project[] = Object.entries(projectsData).map(([id, p]) => ({
  id,
  ...p,
}));
`;

fs.writeFileSync(outPath, header + inner + footer);
console.log('Wrote', outPath, fs.statSync(outPath).size);
