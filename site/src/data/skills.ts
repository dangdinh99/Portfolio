import type { SkillGroup } from './projectTypes';

export const skillGroups: SkillGroup[] = [
  {
    icon: 'fa-code',
    title: 'Programming Languages',
    skills: [
      { name: 'Python', level: 90 },
      { name: 'SQL', level: 85 },
      { name: 'R', level: 75 },
      { name: 'JavaScript', level: 70 },
    ],
  },
  {
    icon: 'fa-brain',
    title: 'ML & Data Science',
    skills: [
      { name: 'Pandas', level: 90 },
      { name: 'Scikit-learn', level: 85 },
      { name: 'XGBoost', level: 82 },
      { name: 'PyTorch', level: 75 },
    ],
  },
  {
    icon: 'fa-tools',
    title: 'Tools & Platforms',
    skills: [
      { name: 'Git', level: 85 },
      { name: 'Snowflake', level: 78 },
      { name: 'AWS', level: 70 },
      { name: 'Docker', level: 68 },
    ],
  },
];

export const expertiseTags: string[] = [
  'Machine Learning',
  'Deep Learning',
  'Exploratory Data Analysis',
  'Data Engineering',
  'NLP',
  'Big Data',
];
