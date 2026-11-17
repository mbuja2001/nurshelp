import React, { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { FiArrowLeft, FiArrowUpRight, FiSearch } from 'react-icons/fi';
import MarketingLayout from '../components/marketing/MarketingLayout';

const stories = [
  {
    slug: 'why-patient-history-needs-to-travel-with-care',
    category: 'South African healthcare',
    title: 'Why patient history needs to travel with care',
    description: 'Fragmented records can leave clinicians without the context they need when a patient moves between services.',
    readTime: '5 min read',
    date: 'A problem statement',
    artwork: 'journal-art-three',
    sections: [
      {
        heading: 'A patient’s story does not fit in one visit',
        paragraphs: [
          'Care in South Africa may involve more than one facility, service or record system. Critical history is not always recorded in a structured way in the current encounter; even when it is documented, it may live in another system that does not share it reliably. The result is not simply an inconvenience: context that matters to clinical assessment can be difficult to locate at the point of care.',
          'Prior encounters, allergies, medicines, chronic conditions and the reasons for earlier laboratory investigations or their results can all help a team understand what is happening now. When that information is missing, hard to search or disconnected from the current encounter, staff may have to reconstruct the story under pressure.'
        ]
      },
      {
        heading: 'The gap affects both decisions and teamwork',
        paragraphs: [
          'Incomplete history can make it harder to interpret a new presentation, compare findings over time or understand what has already been investigated. Teams also need a shared view of the current work: a handover, a consultation and a laboratory follow-up are easier to coordinate when people can return to the relevant patient record.',
          'Chat can support timely staff communication, but it should not become a substitute for the EHR. The durable clinical story belongs in the patient record, where authorized staff can review its source and chronology.'
        ]
      },
      {
        heading: 'A more connected record is a direction, not a claim',
        paragraphs: [
          'NurseHelpDesk exists to help make patient context and care-team workflows more coherent, including in rural clinics, health centres and public hospitals. Unifying fragmented EHR information across facilities is a larger interoperability challenge; it requires compatible systems, trustworthy data exchange and local governance, not a promise that one product can connect every record today.',
          'Our aim is to help preserve a traceable longitudinal history and make useful context easier to review where care happens. Clinicians remain responsible for interpreting that information and deciding what it means for a patient.'
        ]
      }
    ]
  },
  {
    slug: 'longitudinal-context-for-local-ai',
    category: 'Research notes',
    title: 'Research note: longitudinal context for local AI',
    description: 'Our proposed approach explores how structured patient history could support resource-conscious, human-reviewed AI workflows.',
    readTime: '6 min read',
    date: 'Architecture research',
    artwork: 'journal-art-two',
    sections: [
      {
        heading: 'Why patient history is a hard problem for AI',
        paragraphs: [
          'A useful clinical summary may depend on more than the current encounter. Yet sending an entire record to a language model can be costly, difficult to audit and impractical on hardware with limited memory or connectivity. Embedding models and vector databases can add their own compute and infrastructure requirements.',
          'Our research paper proposes a different starting point for structured, timestamped records: use deterministic time windows and clinical rules to assemble a bounded context, then let a model help interpret that context rather than decide which safety-critical history is allowed into it.'
        ]
      },
      {
        heading: 'Keep the history traceable and the clinician in control',
        paragraphs: [
          'The proposed context separates the current encounter, recent events, longer-term trends and critical history. Selected details retain source encounter identifiers and timestamps, while a clinician reviews the context before inference. Deterministic calculations such as triage-score components remain outside the language model.',
          'This direction is intended to make local or resource-conscious inference easier to study. It does not mean that an AI-generated summary is a complete patient record, or that a model should replace the clinician’s review.'
        ]
      },
      {
        heading: 'A research direction, not a clinical claim',
        paragraphs: [
          'The architecture paper describes a proposed design and evaluation approach. It explicitly does not establish clinical validity, diagnostic performance or regulatory approval; prospective evaluation and institutional governance would be required before clinical deployment.',
          'We are exploring how approaches like this could support useful AI in resource-constrained settings, including rural public facilities. Reliable offline operation and safe deployment depend on the complete system, local infrastructure and testing—not on an architecture paper alone.'
        ]
      }
    ]
  },
  {
    slug: 'staff-communication-with-the-ehr-in-view',
    category: 'Care-team coordination',
    title: 'Staff communication works better with the record in view',
    description: 'Connect handovers and staff chat to patient-record context without turning messages into the clinical record.',
    readTime: '4 min read',
    date: 'A workflow perspective',
    artwork: 'journal-art-one',
    sections: [
      {
        heading: 'Care is coordinated by people, not screens',
        paragraphs: [
          'A busy care team needs a practical way to ask for a review, coordinate a handover or clarify who is following up. Staff chat gives colleagues a shared channel for those conversations; the EHR gives them a place to review the patient’s recorded history and encounter information.',
          'When a conversation has no clear patient context, staff may need to search again or repeat details. Making it easier to reference the relevant EHR record helps connect the conversation to the work without asking the chat thread to carry the whole clinical story.'
        ]
      },
      {
        heading: 'Messages are not a substitute for documentation',
        paragraphs: [
          'Chat supports coordination, but it is not the authoritative patient record. Relevant observations, decisions, requests and results belong in the appropriate clinical workflow so they can be traced and reviewed by authorized team members.',
          'That distinction matters for safety and continuity: the next person should be able to find the documented history, not depend on having read every message.'
        ]
      },
      {
        heading: 'Make the workflow fit the facility',
        paragraphs: [
          'Communication patterns differ between clinics, health centres and hospitals. A useful system should support the local team’s roles and handover practices while keeping patient information protected and clinical responsibility clear.',
          'Our goal is to bring staff coordination and EHR context closer together as part of a more coherent care workflow, particularly for public facilities working with constrained resources.'
        ]
      }
    ]
  },
  {
    slug: 'designing-calmer-clinical-handoffs',
    category: 'Perspectives',
    title: 'Designing calmer handoffs in high-pressure environments',
    description: 'Small choices in how information is organized can make transitions between people and teams feel more considered.',
    readTime: '6 min read',
    date: 'A practical perspective',
    artwork: 'journal-art-one',
    sections: [
      {
        heading: 'A handoff is more than a transfer',
        paragraphs: [
          'A clinical handoff is a moment of shared attention. One person is passing context; another is building a picture of what matters next. When that exchange is rushed or fragmented, people spend effort searching for details instead of aligning on the plan.',
          'No interface can remove the complexity of emergency care. Good workflow design can, however, make the important information easier to find and the next step easier to understand.'
        ]
      },
      {
        heading: 'Make the structure do useful work',
        paragraphs: [
          'A consistent sequence helps teams build a shared mental model. Keep identity and current status clear, distinguish observations from interpretation, and make changes easy to trace. Put the most time-sensitive context where it can be found without hunting.',
          'Avoid turning a handoff into a wall of text. Concise sections, meaningful labels and sensible defaults can help preserve attention while still leaving room for the nuance that care requires.'
        ]
      },
      {
        heading: 'Design with the team, then learn from use',
        paragraphs: [
          'The people who give and receive handoffs are best placed to identify friction. Prototype with them, test in realistic scenarios, and listen for workarounds. A polished screen is not proof of a better outcome; the workflow has to make sense in the conditions where it will be used.'
        ]
      }
    ]
  },
  {
    slug: 'human-centered-decision-support',
    category: 'Field notes',
    title: 'What human-centered decision support looks like',
    description: 'Useful digital support brings relevant context forward while keeping professional judgment in the foreground.',
    readTime: '4 min read',
    date: 'A design note',
    artwork: 'journal-art-two',
    sections: [
      {
        heading: 'Support should clarify, not command',
        paragraphs: [
          'Decision-support tools are most useful when they help a clinician see relevant information, understand where it came from and decide what to do with it. They should not disguise uncertainty or present a suggestion as an instruction.',
          'That means making the source and limits of a recommendation visible, designing for review, and giving the clinician a clear path to use their own judgment.'
        ]
      },
      {
        heading: 'Keep context close to the work',
        paragraphs: [
          'When people have to move between screens or systems to assemble a patient picture, important details can become harder to connect. A thoughtful workflow brings relevant history and current encounter information together, while respecting access controls and the local clinical process.',
          'The right amount of context depends on the task. Better organization matters more than simply showing more data.'
        ]
      },
      {
        heading: 'Treat trust as a design requirement',
        paragraphs: [
          'Trust grows when a system behaves predictably, explains itself in plain language and makes correction possible. It should be clear when a tool is unavailable and what the user can do next. These details are not decoration; they shape how technology fits into care.'
        ]
      }
    ]
  },
  {
    slug: 'building-resilient-care-workflows',
    category: 'Operations',
    title: 'Building resilient workflows for a changing care environment',
    description: 'Resilience starts with workflows that help teams adapt when staffing, demand and priorities shift.',
    readTime: '5 min read',
    date: 'An operations brief',
    artwork: 'journal-art-three',
    sections: [
      {
        heading: 'Plan for variation, not an ideal day',
        paragraphs: [
          'Care environments rarely follow a perfect schedule. Staffing changes, arrivals cluster and equipment moves between units. Workflows designed only for the expected path can become brittle as soon as the day changes.',
          'Resilient systems make current assignments visible, allow teams to update operational details and preserve a clear record of what changed.'
        ]
      },
      {
        heading: 'Make ownership and status legible',
        paragraphs: [
          'Teams need to know what is happening, who is responsible and where a task stands. Clear status language and visible ownership can reduce uncertainty without creating more alerts or adding administrative steps.'
        ]
      },
      {
        heading: 'Improve with feedback',
        paragraphs: [
          'Resilience is built over time. Regularly review near misses, friction points and user feedback. Use those observations to refine the workflow, and make sure changes are tested with the people who will rely on them.'
        ]
      }
    ]
  }
];

const categories = ['All stories', ...new Set(stories.map((story) => story.category))];

function JournalArtwork({ className }) {
  return <div className={`journal-art ${className}`} aria-hidden="true"><span /><span /><span /><span /></div>;
}

function BlogIndex() {
  const [category, setCategory] = useState('All stories');
  const [query, setQuery] = useState('');
  const visibleStories = useMemo(() => stories.filter((story) => (
    (category === 'All stories' || story.category === category)
    && `${story.title} ${story.description}`.toLowerCase().includes(query.trim().toLowerCase())
  )), [category, query]);

  return (
    <MarketingLayout>
      <main className="journal-main">
        <header className="journal-hero">
          <span className="marketing-eyebrow">The NurseHelpDesk journal</span>
          <h1>Ideas for care<br /><em>in motion.</em></h1>
          <p>Why NurseHelpDesk exists, the research behind our approach, and practical ideas for connected care in South Africa’s public health facilities.</p>
        </header>
        <div className="journal-toolbar">
          <div className="journal-categories" aria-label="Filter stories by topic">
            {categories.map((item) => (
              <button
                type="button"
                key={item}
                className={`journal-category${category === item ? ' is-selected' : ''}`}
                aria-pressed={category === item}
                onClick={() => setCategory(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <label className="journal-search">
            <FiSearch aria-hidden="true" />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search stories" aria-label="Search stories" />
          </label>
        </div>
        <section className="journal-grid" aria-live="polite">
          {visibleStories.map((story) => (
            <article className="journal-card" key={story.slug}>
              <JournalArtwork className={story.artwork} />
              <div className="journal-card-body">
                <div className="journal-preview-meta">{story.category}<span>•</span>{story.readTime}</div>
                <h2>{story.title}</h2>
                <p>{story.description}</p>
                <Link className="text-link" to={`/blog/${story.slug}`}>Read the story <FiArrowUpRight aria-hidden="true" /></Link>
              </div>
            </article>
          ))}
          {!visibleStories.length && <p className="journal-empty">No stories match that search. Try another title or topic.</p>}
        </section>
      </main>
    </MarketingLayout>
  );
}

function BlogArticle({ story }) {
  return (
    <MarketingLayout>
      <main className="journal-main">
        <article className="journal-article">
          <Link className="journal-back" to="/blog"><FiArrowLeft aria-hidden="true" /> Back to the journal</Link>
          <span className="marketing-eyebrow">{story.category}</span>
          <h1>{story.title}</h1>
          <p className="journal-article-deck">{story.description}</p>
          <div className="journal-article-meta">{story.date} <span>•</span> {story.readTime}</div>
          <JournalArtwork className={story.artwork} />
          <div className="journal-article-body">
            {story.sections.map((section, index) => (
              <React.Fragment key={section.heading}>
                <h2>{section.heading}</h2>
                {section.paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
                {index === 0 && <aside className="journal-article-callout">Technology can make information easier to work with. Clinical decisions belong to qualified care professionals.</aside>}
              </React.Fragment>
            ))}
          </div>
        </article>
      </main>
    </MarketingLayout>
  );
}

export default function Blog() {
  const { slug } = useParams();
  if (!slug) return <BlogIndex />;
  const story = stories.find((item) => item.slug === slug);
  if (!story) {
    return (
      <MarketingLayout>
        <main className="journal-main journal-article">
          <Link className="journal-back" to="/blog"><FiArrowLeft aria-hidden="true" /> Back to the journal</Link>
          <h1>Story not found</h1>
          <p className="journal-article-deck">That article may have moved. Browse the journal to find another story.</p>
        </main>
      </MarketingLayout>
    );
  }
  return <BlogArticle story={story} />;
}
