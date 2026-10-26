// JARVIS AI - Core JavaScript
const pieces=['♜','♞','♝','♛','♚','♝','♞','♜'];
const pawns='♟';const wPieces=['♖','♘','♗','♕','♔','♗','♘','♖'];const wPawn='♙';
let recognition,isListening=false,currentPage='home';

function showDashboard(){
  document.getElementById('landing-page').style.display='none';
  document.getElementById('dashboard-page').style.display='flex';
  document.getElementById('main-nav').style.display='none';
  startLiveFeed();
}
function showLanding(){
  document.getElementById('landing-page').style.display='block';
  document.getElementById('dashboard-page').style.display='none';
  document.getElementById('main-nav').style.display='flex';
}
function toggleMobileMenu(){document.getElementById('mobile-menu').classList.toggle('open')}
function toggleSidebar(){document.getElementById('sidebar').classList.toggle('expanded')}
function switchPage(page){
  currentPage=page;
  document.querySelectorAll('.dash-page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+page).classList.add('active');
  document.querySelectorAll('.sidebar-item').forEach(i=>i.classList.remove('active'));
  document.querySelector('[data-page="'+page+'"]').classList.add('active');
  const titles={home:'COMMAND CENTER',games:'GAMES ARENA',booking:'BOOKING AGENT',writing:'WRITING STUDIO',memory:'MEMORY CORE',settings:'SETTINGS'};
  document.getElementById('page-title').textContent=titles[page]||'JARVIS';
}

// Chess
let board=Array(8).fill(null).map(()=>Array(8).fill(''));
let selectedSq=null,playerColor='w',moveHistory=[];
function initBoard(){
  for(let i=0;i<8;i++){board[0][i]=pieces[i];board[1][i]=pawns;board[6][i]=wPawn;board[7][i]=wPieces[i];}
}
function renderBoard(containerId){
  const container=document.getElementById(containerId);if(!container)return;
  container.innerHTML='';
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const sq=document.createElement('div');
    sq.className='chess-sq '+((r+c)%2===0?'light':'dark');
    sq.textContent=board[r][c];sq.style.fontSize='1.4rem';sq.style.cursor='pointer';
    sq.dataset.r=r;sq.dataset.c=c;
    if(selectedSq&&selectedSq.r===r&&selectedSq.c===c)sq.style.boxShadow='inset 0 0 10px var(--gold-pure)';
    sq.onclick=()=>handleSquareClick(r,c);
    container.appendChild(sq);
  }
}
function handleSquareClick(r,c){
  if(selectedSq){
    const from=selectedSq;
    if(board[from.r][from.c]){
      board[r][c]=board[from.r][from.c];board[from.r][from.c]='';
      const cols='abcdefgh';
      moveHistory.push(cols[from.c]+(8-from.r)+' → '+cols[c]+(8-r));
      updateMoveLog();selectedSq=null;renderBoard('chess-board');
      setTimeout(jarvisMove,800);
    }
  }else if(board[r][c]){selectedSq={r,c};renderBoard('chess-board');}
}
function jarvisMove(){
  document.getElementById('thinking').style.display='block';
  setTimeout(()=>{
    let moves=[];
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){
      if(pieces.includes(board[r][c])||board[r][c]===pawns){
        for(let tr=0;tr<8;tr++)for(let tc=0;tc<8;tc++){
          if(board[tr][tc]===''||(wPieces.includes(board[tr][tc])||board[tr][tc]===wPawn))
            moves.push({fr:r,fc:c,tr,tc});
        }
      }
    }
    if(moves.length){
      const m=moves[Math.floor(Math.random()*moves.length)];
      board[m.tr][m.tc]=board[m.fr][m.fc];board[m.fr][m.fc]='';
      const cols='abcdefgh';
      moveHistory.push('Jarvis: '+cols[m.fc]+(8-m.fr)+' → '+cols[m.tc]+(8-m.tr));
      updateMoveLog();
    }
    document.getElementById('thinking').style.display='none';
    renderBoard('chess-board');
  },1000);
}
function updateMoveLog(){
  const log=document.getElementById('move-log');if(!log)return;
  log.innerHTML=moveHistory.map(m=>'<div class="log-line">> '+m+'</div>').join('');
  log.scrollTop=log.scrollHeight;
}
function startChess(){
  initBoard();
  document.getElementById('chess-area').style.display='flex';
  renderBoard('chess-board');
}

// Live Feed
const feedMessages=[
  '> Voice input received','> Intent classified: general.query','> Processing natural language...','> Response generated in 0.28s',
  '> Memory core updated','> Context window: 12 interactions','> Voice synthesis: speaking response','> Idle — awaiting command',
  '> System check: all modules online','> Chess engine: standby','> Booking API: connected','> Writing model: loaded'
];
let feedInterval;
function startLiveFeed(){
  const log=document.getElementById('live-log');if(!log)return;
  const now=new Date();
  log.innerHTML='<div class="log-line">> '+formatTime(now)+' JARVIS system initialized</div>';
  let i=0;
  feedInterval=setInterval(()=>{
    const t=new Date();
    const line=document.createElement('div');line.className='log-line';
    line.textContent='> '+formatTime(t)+' '+feedMessages[i%feedMessages.length].substring(2);
    log.appendChild(line);log.scrollTop=log.scrollHeight;
    if(log.children.length>50)log.removeChild(log.firstChild);
    i++;
  },3000);
}
function formatTime(d){return d.toTimeString().split(' ')[0]}

// Voice
function toggleVoice(){
  const btn=document.getElementById('mic-btn');
  if(!isListening){
    try{
      recognition=new(window.SpeechRecognition||window.webkitSpeechRecognition)();
      recognition.continuous=false;recognition.lang='en-IN';
      recognition.onresult=(e)=>{
        const cmd=e.results[0][0].transcript;
        document.getElementById('command-input').value=cmd;
        routeCommand(cmd);
      };
      recognition.onend=()=>{isListening=false;btn.classList.remove('active');
        const wf=document.getElementById('waveform');if(wf)wf.classList.remove('active');};
      recognition.start();isListening=true;btn.classList.add('active');
      const wf=document.getElementById('waveform');if(wf)wf.classList.add('active');
    }catch(e){alert('Voice not supported in this browser')}
  }else{recognition.stop();isListening=false;btn.classList.remove('active')}
}
function routeCommand(text){
  const t=text.toLowerCase();
  if(t.includes('chess')||t.includes('game'))switchPage('games');
  else if(t.includes('book')||t.includes('flight')||t.includes('hotel'))switchPage('booking');
  else if(t.includes('write')||t.includes('email')||t.includes('essay'))switchPage('writing');
  else if(t.includes('home'))switchPage('home');
  else if(t.includes('memory'))switchPage('memory');
  else if(t.includes('setting'))switchPage('settings');
  addLogLine('Command: '+text);
}
function addLogLine(msg){
  const log=document.getElementById('live-log');if(!log)return;
  const line=document.createElement('div');line.className='log-line';
  line.textContent='> '+formatTime(new Date())+' '+msg;
  log.appendChild(line);log.scrollTop=log.scrollHeight;
}
function addBooking(){
  const input=document.getElementById('booking-input');
  if(input.value.trim()){addLogLine('Booking request: '+input.value);input.value='';alert('Jarvis is processing your booking request...');}
}

// Chess display for landing page
function renderDisplayBoard(){
  const container=document.getElementById('chess-display');if(!container)return;
  const displayBoard=[
    ['♜','♞','♝','♛','♚','♝','♞','♜'],['♟','♟','♟','♟','♟','♟','♟','♟'],
    ['','','','','','','',''],['','','','♟','','','',''],
    ['','','','','♙','','',''],['','','','','','','',''],
    ['♙','♙','♙','♙','','♙','♙','♙'],['♖','♘','♗','♕','♔','♗','♘','♖']
  ];
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const sq=document.createElement('div');
    sq.className='chess-sq '+((r+c)%2===0?'light':'dark');
    sq.textContent=displayBoard[r][c];sq.style.fontSize='1rem';
    container.appendChild(sq);
  }
}

// Keyboard shortcut
document.addEventListener('keydown',(e)=>{
  if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();
    const input=document.getElementById('command-input');if(input)input.focus();}
});

// Init
document.addEventListener('DOMContentLoaded',()=>{renderDisplayBoard();});
