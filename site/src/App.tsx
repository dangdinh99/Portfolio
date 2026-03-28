import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { About } from './components/About';
import { ProjectsSection } from './components/ProjectsSection';
import { ContestsSection } from './components/ContestsSection';
import { Experience } from './components/Experience';
import { Certifications } from './components/Certifications';
import { Contact } from './components/Contact';
import { Footer } from './components/Footer';

function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <About />
      <ProjectsSection />
      <ContestsSection />
      <Experience />
      <Certifications />
      <Contact />
      <Footer />
    </>
  );
}

export default App;
