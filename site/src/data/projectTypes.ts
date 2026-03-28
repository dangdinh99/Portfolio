export interface ProjectStat {
  value: string;
  label: string;
}

export interface Project {
  id: string;
  title: string;
  category: string;
  image: string;
  description: string;
  longDescription: string;
  tech: string[];
  features: string[];
  stats?: ProjectStat[];
  github?: string | null;
  demo?: string | null;
  report?: string | null;
  hasGithub: boolean;
  hasDemo: boolean;
  hasReport?: boolean;
}

export interface Contest {
  id: string;
  title: string;
  organizer: string;
  date: string;
  result: string;
  description: string;
  tech: string[];
  image?: string;
  link?: string;
  longDescription?: string;
}
