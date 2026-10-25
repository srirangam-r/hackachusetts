import React from 'react';
import {AbsoluteFill, Img, interpolate, OffthreadVideo, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {loadFont} from '@remotion/google-fonts/AtkinsonHyperlegibleNext';
import {loadFont as loadMono} from '@remotion/google-fonts/AtkinsonHyperlegibleMono';
import results from './results.json';

const {fontFamily: SANS} = loadFont();
const {fontFamily: MONO} = loadMono();
const C = {bg: '#e9edf0', surface: '#f8f9fa', ink: '#16202a', ink2: '#56626e', rule: '#d2d9df', screen: '#0d1217',
  brain: '#5a43c8', brainSoft: '#ebe7fb', demo: '#0a7479', demoSoft: '#dcefef', geo: '#9c5c00', bad: '#b3322a', ok: '#1d7443'};
const f = (s: string) => staticFile(`footage/${s}`);

// scene lengths in frames (30 fps)
const SCENES = {title: 150, problem: 270, layers: 360, hero: 1020, ur5e: 300, edit: 270, results: 390, kiro: 660, close: 180};
export const TOTAL = Object.values(SCENES).reduce((a, b) => a + b, 0);

const useIn = (delay = 0) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: {damping: 200}});
};
const Fade: React.FC<{delay?: number; children: React.ReactNode; style?: React.CSSProperties}> = ({delay = 0, children, style}) => {
  const p = useIn(delay);
  return <div style={{opacity: p, transform: `translateY(${(1 - p) * 24}px)`, ...style}}>{children}</div>;
};
const Page: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <AbsoluteFill style={{background: C.bg, color: C.ink, fontFamily: SANS, padding: '90px 120px', ...style}}>{children}</AbsoluteFill>
);
const H: React.FC<{children: React.ReactNode; size?: number}> = ({children, size = 76}) => (
  <h1 style={{fontSize: size, lineHeight: 1.08, fontWeight: 700, margin: 0, letterSpacing: '-0.015em'}}>{children}</h1>
);
const Sub: React.FC<{children: React.ReactNode}> = ({children}) => (
  <p style={{fontSize: 38, lineHeight: 1.3, color: C.ink2, margin: '22px 0 0', maxWidth: 1400}}>{children}</p>
);
const Frame: React.FC<{children: React.ReactNode; color?: string; style?: React.CSSProperties}> = ({children, color = C.screen, style}) => (
  <div style={{borderRadius: 14, overflow: 'hidden', background: C.screen, outline: `5px solid ${color}`, ...style}}>{children}</div>
);

// ---------------------------------------------------------------------------- scenes
const Title = () => (
  <Page style={{flexDirection: 'row', alignItems: 'center', gap: 80}}>
    <div style={{flex: 1}}>
      <Fade><H size={110}>Motor Memory</H></Fade>
      <Fade delay={12}><Sub>A robot that plans with Kiro and moves by copying human demos.</Sub></Fade>
      <Fade delay={24}><p style={{fontSize: 30, color: C.ink2, marginTop: 40}}>Hackachusetts, Robotics & Hardware track</p></Fade>
    </div>
    <Frame style={{width: 760, height: 760}}><OffthreadVideo src={f('sim_hero.mp4')} muted style={{width: '100%', height: '100%'}} /></Frame>
  </Page>
);

const Stat: React.FC<{who: string; v: string; delay: number; color?: string}> = ({who, v, delay, color = C.bad}) => (
  <Fade delay={delay} style={{flex: 1, borderTop: `6px solid ${color}`, paddingTop: 24}}>
    <div style={{fontSize: 120, fontWeight: 700, color}}>{v}</div>
    <div style={{fontSize: 36, color: C.ink2}}>{who}</div>
  </Fade>
);
const Problem = () => (
  <Page>
    <Fade><H>Robot foundation models memorize their training scenes.</H></Fade>
    <Fade delay={15}><Sub>They score 79 to 98% on LIBERO-Goal. Give them a new instruction in the same scenes (LIBERO-Pro):</Sub></Fade>
    <div style={{display: 'flex', gap: 60, marginTop: 70}}>
      <Stat who="OpenVLA" v="0%" delay={40} />
      <Stat who="π0" v="0%" delay={55} />
      <Stat who="π0.5" v="≈2%" delay={70} />
    </div>
    <Fade delay={120}><Sub>Frontier models reason far better, but they can't move a robot precisely. So split the job.</Sub></Fade>
  </Page>
);

const Strip: React.FC<{src: string; n: number; size: number}> = ({src, n, size}) => {
  const frame = useCurrentFrame();
  const k = Math.floor(frame / 3) % n;
  return <div style={{width: size, height: size, backgroundImage: `url(${src})`, backgroundSize: `${n * 100}% 100%`,
    backgroundPosition: `${(k / (n - 1)) * 100}% 0`}} />;
};
const Layer: React.FC<{title: string; color: string; delay: number; children: React.ReactNode; note: string}> = ({title, color, delay, children, note}) => (
  <Fade delay={delay} style={{flex: 1, display: 'flex', flexDirection: 'column', gap: 22}}>
    <div style={{fontSize: 44, fontWeight: 700, color}}>{title}</div>
    <Frame color={color} style={{width: 470, height: 470, display: 'grid', placeItems: 'center'}}>{children}</Frame>
    <div style={{fontSize: 28, color: C.ink2, lineHeight: 1.35, maxWidth: 480}}>{note}</div>
  </Fade>
);
const Layers = () => (
  <Page style={{padding: '70px 120px'}}>
    <Fade><H size={64}>Three layers, each doing one job</H></Fade>
    <div style={{display: 'flex', gap: 60, marginTop: 50}}>
      <Layer title="Kiro decides what" color={C.brain} delay={20} note="Reads the task and two marked images, writes a 1 to 3 line program. Never code.">
        <pre style={{fontFamily: MONO, fontSize: 40, color: '#e5eaef', margin: 0, lineHeight: 1.6}}>{'pick(4)\nplace(3, "in")'}</pre>
      </Layer>
      <Layer title="SAM 3 finds where" color={C.ink} delay={80} note="Numbered masks plus depth give every object a 3-D position and frame.">
        <Img src={f('marked_front.png')} style={{width: '100%', height: '100%'}} />
      </Layer>
      <Layer title="Human demos show how" color={C.demo} delay={140} note="9,281 motions from 4,200 recorded demos, stored relative to the object they act on.">
        <Strip src={f('demo_butter.jpg')} n={40} size={470} />
      </Layer>
    </div>
    <Fade delay={210}><p style={{fontSize: 34, marginTop: 34}}>The demo motion is re-anchored on the live object and run with operational-space control.</p></Fade>
  </Page>
);

// the live page, filmed from a recorded run (2x), with zooms and captions
const HERO_CAPS: [number, number, string][] = [
  [0, 120, 'The live demo page. Task typed in plain English; Kiro looks first.'],
  [120, 330, 'Every motion shows the human demo it copied, each recorded for a different task.'],
  [330, 420, 'Kiro sees only the camera images and the marks. No simulator state.'],
  [420, 640, 'Skills check themselves: the orange juice grasp missed, so Kiro looks again and retries.'],
  [640, 1020, 'Done: drawer opened, butter inside, drawer closed, orange juice in the tray. 67 s, 8 Kiro calls.'],
];
const Hero = () => {
  const frame = useCurrentFrame();
  // zoom keyframes: [frame, scale, focus x, focus y]
  const keys: [number, number, number, number][] = [[0, 1, 960, 540], [110, 1, 960, 540], [140, 1.6, 1900, 830],
    [300, 1.6, 1900, 830], [330, 1.5, 1900, 420], [410, 1.5, 1900, 420], [440, 1.45, 1020, 940], [620, 1.45, 1020, 940],
    [650, 1, 960, 540], [1020, 1, 960, 540]];
  const at = (i: number) => interpolate(frame, keys.map((k) => k[0]), keys.map((k) => k[i]), {extrapolateRight: 'clamp'});
  const s = at(1), x = at(2), y = at(3);
  const cap = HERO_CAPS.find(([a, b]) => frame >= a && frame < b);
  return (
    <AbsoluteFill style={{background: C.bg, fontFamily: SANS}}>
      <AbsoluteFill style={{transform: `scale(${s})`, transformOrigin: `${x}px ${y}px`}}>
        {frame < 895 ? <OffthreadVideo src={f('page_ep002.mp4')} muted />
          : <Img src={f('page_ep002/00599.jpg')} style={{width: 1920, height: 1080}} />}
      </AbsoluteFill>
      {cap && (
        <div style={{position: 'absolute', left: 60, right: 60, top: 30, background: 'rgba(22,32,42,0.92)', color: '#fff',
          fontSize: 40, fontWeight: 600, padding: '20px 30px', borderRadius: 12, lineHeight: 1.3}}>{cap[2]}</div>
      )}
    </AbsoluteFill>
  );
};

const Clip: React.FC<{title: string; text: string; video: string; extra?: React.ReactNode; rate?: number}> = ({title, text, video, extra, rate = 1}) => (
  <Page style={{flexDirection: 'row', alignItems: 'center', gap: 80}}>
    <div style={{flex: 1}}>
      <Fade><H size={68}>{title}</H></Fade>
      <Fade delay={15}><Sub>{text}</Sub></Fade>
      {extra}
    </div>
    <Frame style={{width: 760, height: 760}}><OffthreadVideo src={f(video)} muted playbackRate={rate} style={{width: '100%', height: '100%'}} /></Frame>
  </Page>
);
const Ur5e = () => (
  <Clip title="Same demos, different robot" video="ur5e_wine.mp4"
    text="This UR5e has no demos of its own. It puts the wine bottle on the cabinet using motions recorded on a Panda."
    extra={<Fade delay={40}>
      <div style={{fontSize: 28, lineHeight: 1.45, marginTop: 40, color: C.demo}}>
        <div><b>Grasp</b> copied from a Panda demo of “put the wine bottle on the rack”</div>
        <div><b>Place</b> copied from a Panda demo of “put the wine bottle on top of the cabinet”</div>
      </div>
      <p style={{fontSize: 30, marginTop: 30}}>On LIBERO-Goal the UR5e keeps most of the Panda's success.</p>
    </Fade>} />
);
const Status: React.FC<{ok: boolean; text: string; label: string; delay: number}> = ({ok, text, label, delay}) => (
  <Fade delay={delay} style={{flex: 1}}>
    <div style={{fontSize: 30, fontWeight: 700, color: ok ? C.demo : C.bad, marginBottom: 14}}>{label}</div>
    <pre style={{fontFamily: MONO, fontSize: 32, lineHeight: 1.45, background: C.surface, border: `3px solid ${ok ? C.demo : C.bad}`,
      borderRadius: 12, padding: 28, margin: 0, whiteSpace: 'pre-wrap', color: ok ? C.demo : C.bad}}>{text}</pre>
  </Fade>
);
const Edit = () => (
  <Page style={{justifyContent: 'center'}}>
    <Fade><H size={72}>Delete a demo, the skill is gone. Add it back, it returns.</H></Fade>
    <div style={{display: 'flex', gap: 50, marginTop: 60}}>
      <Status ok={false} delay={25} label="Black book demos removed from motor memory"
        text={'pick(5) -> NO_CANDIDATE: no demo\nof black book in motor memory'} />
      <Status ok delay={90} label="Demos restored"
        text={'pick(5) -> OK: copied from a\nbook demo (STUDY_SCENE3, demo 48)'} />
    </div>
    <Fade delay={140}><Sub>Every motion traces back to a demo id. Teaching a new object means recording one demo, not retraining a model.</Sub></Fade>
  </Page>
);
const Slope = () => {
  const p = useIn(25);
  const W = 1080, H = 620, top = 30, bot = 520;
  const xs = [150, 460, 770];
  const y = (v: number) => top + ((100 - v) / 100) * (bot - top);
  const pts = (vals: number[]) => vals.map((v, i) => `${xs[i]},${y(v)}`).join(' ');
  const band = `${pts(results.vla_top)} ${results.vla_bottom.map((v, i) => `${xs[i]},${y(v)}`).reverse().join(' ')}`;
  const clipW = xs[0] - 20 + (xs[2] - xs[0] + 40) * p;
  const last = xs.length - 1;
  return (
    <svg width={W} height={H} style={{overflow: 'visible', fontFamily: SANS}}>
      <defs><clipPath id="reveal"><rect x={0} y={0} width={clipW} height={H} /></clipPath></defs>
      {[0, 50, 100].map((v) => (
        <g key={v}>
          <line x1={xs[0] - 30} x2={xs[last] + 30} y1={y(v)} y2={y(v)} stroke={C.rule} strokeWidth={2} />
          <text x={xs[0] - 44} y={y(v) + 9} fontSize={26} fill={C.ink2} textAnchor="end">{v}%</text>
        </g>
      ))}
      <g clipPath="url(#reveal)">
        <polygon points={band} fill="#9aa6b1" fillOpacity={0.3} stroke="#7d8893" strokeWidth={4} strokeLinejoin="round" />
        <polyline points={pts(results.ours)} fill="none" stroke={C.demo} strokeWidth={8} strokeLinejoin="round" strokeLinecap="round" />
        {results.ours.map((v, i) => <circle key={i} cx={xs[i]} cy={y(v)} r={11} fill={C.demo} />)}
      </g>
      <g opacity={interpolate(p, [0.8, 1], [0, 1], {extrapolateLeft: 'clamp'})}>
        <text x={xs[last] + 26} y={y(results.ours[last]) + 10} fontSize={32} fontWeight={700} fill={C.demo}>Motor Memory</text>
        <text x={xs[last] + 26} y={y(results.vla_top[last]) + 22} fontSize={32} fontWeight={700} fill="#56626e">Fine-tuned VLAs</text>
      </g>
      {results.conditions.map((c, i) => (
        <text key={c} x={xs[i]} y={bot + 60} fontSize={30} fontWeight={600} fill={C.ink} textAnchor="middle">{c}</text>
      ))}
    </svg>
  );
};
const Results = () => (
  <Page style={{padding: '70px 110px'}}>
    <Fade><H size={66}>VLAs win on what they trained on. We hold up when the scene changes.</H></Fade>
    <div style={{display: 'flex', gap: 40, marginTop: 50, alignItems: 'flex-start'}}>
      <Slope />
      <div style={{flex: 1, display: 'grid', gap: 34, paddingTop: 10}}>
        <Fade delay={70}>
          <div style={{fontSize: 32, fontWeight: 700}}>On familiar tasks, VLAs are ahead.</div>
          <div style={{fontSize: 27, color: C.ink2, lineHeight: 1.35, marginTop: 8}}>Our gap sits in parts we can improve one at a time: grasps, drawer handles, insertion, perception. No retraining.</div>
        </Fade>
        <Fade delay={130}>
          <div style={{fontSize: 32, fontWeight: 700, color: C.demo}}>When the scene changes, we hold up.</div>
          <div style={{fontSize: 27, color: C.ink2, lineHeight: 1.35, marginTop: 8}}>Moved objects: on par with the best VLA, while the other two fall to zero. New instructions: every VLA falls to about zero; we still solve about a quarter. No practice on the test tasks.</div>
        </Fade>
      </div>
    </div>
    <Fade delay={180}><p style={{fontSize: 20, color: C.ink2, marginTop: 18}}>VLAs (OpenVLA, π0, π0.5): published LIBERO-Goal and LIBERO-Pro results. Ours: 50 episodes per condition at LIBERO-Pro's 300-step limit, Kiro brain, SAM 3 perception.</p></Fade>
  </Page>
);
const KIRO: [string, string, string][] = [
  ['Spec-driven', 'requirements → design → tasks. The interfaces in design.md let parallel Kiro agents build 17 stream branches in git worktrees.', '- [x] 0. Contracts and shared geometry\n- [x] 1. Stream A: env\n- [x] 5. Stream E: brain'],
  ['Steering', 'fairness.md is always included: what the brain may never use. tech.md holds verified simulator facts.', 'At test time the robot brain and skills\nmust never use: simulator object state,\nthe success signal, sim masks ...'],
  ['Hooks', 'A PreToolUse guard blocks force push, reset --hard and data deletion, which makes unattended -a runs safe. Tests run on save.', '"trigger": "PreToolUse",\n"matcher": "execute_bash",\n"command": "python3 scripts/hooks/guard.py"'],
  ['Property-based tests', 'Hypothesis found a real bug: 17 points over 4.3 cm with 0.7 mm noise gave a 3.15° axis error against a fixed 3° bound.', 'bound ∝ noise / (length · √n)'],
  ['MCP', 'Our robot MCP server (observe, run, done) is the brain\'s only interface. alphaXiv MCP for 40+ papers.', 'tools: ["@robot"]\nincludeMcpJson: false'],
  ['Kiro is the runtime brain', 'Headless kiro-cli runs every episode as a custom agent; a custom power packages the harness for any Kiro agent.', 'kiro-cli chat --no-interactive -a\n  --agent robot-brain --model <M>'],
];
const Kiro = () => (
  <Page style={{padding: '60px 90px'}}>
    <Fade><H size={64}>Built with Kiro, and run by Kiro</H></Fade>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 28, marginTop: 36}}>
      {KIRO.map(([t, d, code], i) => (
        <Fade key={t} delay={20 + i * 55} style={{background: C.surface, border: `2px solid ${C.rule}`, borderRadius: 14, padding: 26,
          borderTop: `6px solid ${C.brain}`}}>
          <div style={{fontSize: 34, fontWeight: 700, color: C.brain}}>{t}</div>
          <div style={{fontSize: 24, lineHeight: 1.35, marginTop: 10}}>{d}</div>
          <pre style={{fontFamily: MONO, fontSize: 19, background: C.brainSoft, borderRadius: 8, padding: 14, marginTop: 14,
            whiteSpace: 'pre-wrap', marginBottom: 0}}>{code}</pre>
        </Fade>
      ))}
    </div>
    <Fade delay={420}><p style={{fontSize: 22, color: C.ink2, marginTop: 24}}>
      The live demo page and this video were made with Claude Code after our Kiro credits ran out; the robot's brain is still Kiro.</p></Fade>
  </Page>
);
const Close = () => (
  <Page style={{justifyContent: 'center'}}>
    <Fade><H size={110}>Motor Memory</H></Fade>
    <Fade delay={12}><Sub>Kiro decides what. SAM 3 finds where. Human demos show how.</Sub></Fade>
    <Fade delay={30}><p style={{fontFamily: MONO, fontSize: 40, marginTop: 50}}>github.com/srirangam-r/hackachusetts</p></Fade>
  </Page>
);

export const Video = () => {
  const order: [keyof typeof SCENES, React.FC][] = [['title', Title], ['problem', Problem], ['layers', Layers], ['hero', Hero],
    ['ur5e', Ur5e], ['edit', Edit], ['results', Results], ['kiro', Kiro], ['close', Close]];
  let t = 0;
  return (
    <AbsoluteFill style={{background: C.bg}}>
      {order.map(([k, S]) => {
        const from = t;
        t += SCENES[k];
        return <Sequence key={k} from={from} durationInFrames={SCENES[k]}><S /></Sequence>;
      })}
    </AbsoluteFill>
  );
};
