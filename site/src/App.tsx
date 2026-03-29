import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { About } from './components/About';
import { ProjectsSection } from './components/ProjectsSection';
import { Experience } from './components/Experience';
import { Contact } from './components/Contact';
import { Footer } from './components/Footer';

function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <About />
      <ProjectsSection />
      <Experience />
      <Contact />
      <Footer />
    </>
  );
}

export default App;
