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

export interface Certification {
  iconClass: string;
  title: string;
  provider: string;
  date: string;
  badge: string;
}

export interface ExperienceItem {
  date: string;
  title: string;
  company: string;
  icon: string;
  isSchool: boolean;
  description: string;
}

export interface Skill {
  name: string;
  level: number;
}

export interface SkillGroup {
  icon: string;
  title: string;
  skills: Skill[];
}

export interface HeroStat {
  value: number;
  suffix: string;
  label: string;
  icon: string;
}
