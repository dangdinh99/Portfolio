import type { Contest } from './projectTypes';

export const contests: Contest[] = [
  {
    id: 'contest1',
    title: 'Boston University Spark! Data Science',
    organizer: 'BU Spark! / Campaign Zero',
    date: '2024–2025',
    result: 'Team contributor — 911 ETL pipeline',
    description:
      'Collaborated on a multi-city 911 data harmonization pipeline delivering unified analytics for public safety research.',
    tech: ['Python', 'Snowflake', 'GeoPandas', 'YAML'],
    image: 'image/campaignzero.png',
    link: 'https://github.com/BU-Spark/ds-cz-911',
    longDescription:
      '<p>Worked with BU Spark! and Campaign Zero on ingesting and standardizing large-scale 911 call data across cities. Focused on scalable ETL and geographic enrichment.</p>',
  },
  {
    id: 'contest2',
    title: 'Kaggle-style ML challenge (stroke risk)',
    organizer: 'Course / portfolio',
    date: '2024',
    result: '94% AUC — ensemble models',
    description:
      'End-to-end stroke risk modeling with engineered indices, clustering, and supervised learners with full evaluation.',
    tech: ['Python', 'Scikit-learn', 'XGBoost'],
    image: 'image/stroke.jpg',
    link: 'https://github.com/dangdinh99/health_prediction_challenge',
  },
];
