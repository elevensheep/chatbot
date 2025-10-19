import React, { useState, useRef } from 'react';
import './App.css';
import Header from './components/Header/Header';
import ChatWindow from './components/ChatWindow/ChatWindow';
import Login from './components/Login/Login';
import Signup from './components/Signup/Signup';
import Sidebar from './components/Sidebar/Sidebar';
import RightToolbar from './components/RightToolbar/RightToolbar';
import Footer from './components/Footer/Footer';
import StudyPlan from './components/StudyPlan/StudyPlan';


let idCounter = 1;
function makeDefaultSession() {
  return {
    id: Date.now(),
    messages: [
      { id: idCounter++, from: 'bot', text: '안녕하세요! 무엇을 도와드릴까요?', ts: Date.now() },
    ],
    created: Date.now(),
  };
}

function App() {
  // 상태 관리
  const [sessions, setSessions] = useState([makeDefaultSession()]);
  const [currentSessionIdx, setCurrentSessionIdx] = useState(0);
  const currentSessionIdxRef = useRef(currentSessionIdx);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pendingSession, setPendingSession] = useState(null);
  const [user, setUser] = useState(null);
  const [showSignup, setShowSignup] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [currentPage, setCurrentPage] = useState('chat'); // 페이지 상태 추가

  React.useEffect(() => {
    currentSessionIdxRef.current = currentSessionIdx;
  }, [currentSessionIdx]);

  async function handleSend(text) {
    if (pendingSession) {
      const newSession = {
        ...pendingSession,
        messages: [
          ...pendingSession.messages,
          { id: idCounter++, from: 'user', text, ts: Date.now() }
        ]
      };
      setSessions(prev => {
        const newSessionIdx = prev.length;
        const updated = [...prev, newSession];
        setCurrentSessionIdx(newSessionIdx);
        setPendingSession(null);
        
        // 백엔드 API 호출
        callBackendAPI(text, newSessionIdx);
        
        return updated;
      });
      return;
    }
    
    const idx = currentSessionIdxRef.current;
    setSessions(prev => {
      const updated = prev.map((session, i) =>
        i === idx
          ? { ...session, messages: [...session.messages, { id: idCounter++, from: 'user', text, ts: Date.now() }] }
          : session
      );
      return updated;
    });
    
    // 백엔드 API 호출
    callBackendAPI(text, idx);
  }
  
  // 백엔드 API 호출 함수
  async function callBackendAPI(text, sessionIdx) {
    const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
    
    console.log('🚀 백엔드 API 호출 시작:', text);
    console.log('📡 API URL:', API_URL);
    
    try {
      // 로딩 메시지 추가
      setSessions(prev => {
        const updated = [...prev];
        const loadingMsg = { 
          id: idCounter++, 
          from: 'bot', 
          text: '답변을 생성 중입니다...', 
          ts: Date.now(),
          isLoading: true 
        };
        updated[sessionIdx].messages = [...updated[sessionIdx].messages, loadingMsg];
        return updated;
      });
      
      // 백엔드 API 호출 (/chat 엔드포인트 - LLM 답변 생성)
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: text,
          top_k: 3,
          return_sources: true  // 출처 정보도 함께 반환
        })
      });
      
      if (!response.ok) {
        throw new Error(`API 오류: ${response.status}`);
      }
      
      const data = await response.json();
      
      console.log('✅ API 응답 받음:', data);
      
      // LLM이 생성한 자연스러운 답변 사용
      let answerText = data.answer || '죄송합니다. 답변을 생성할 수 없습니다.';
      
      // (선택사항) 출처 정보 추가
      if (data.sources && data.sources.length > 0) {
        answerText += '\n\n📚 참고한 수업계획서:\n';
        data.sources.forEach((source, index) => {
          const subject = source.metadata?.subject || '알 수 없음';
          const type = source.metadata?.type || '';
          answerText += `${index + 1}. ${subject}`;
          if (type) answerText += ` (${type})`;
          answerText += '\n';
        });
      }
      
      // 로딩 메시지를 실제 답변으로 교체
      setSessions(prev => {
        const updated = [...prev];
        const messages = updated[sessionIdx].messages;
        // 마지막 로딩 메시지 제거
        const filteredMessages = messages.filter(msg => !msg.isLoading);
        // 실제 답변 추가
        const botMsg = { 
          id: idCounter++, 
          from: 'bot', 
          text: answerText, 
          ts: Date.now() 
        };
        updated[sessionIdx].messages = [...filteredMessages, botMsg];
        return updated;
      });
      
    } catch (error) {
      console.error('API 호출 오류:', error);
      
      // 오류 메시지 표시
      setSessions(prev => {
        const updated = [...prev];
        const messages = updated[sessionIdx].messages;
        const filteredMessages = messages.filter(msg => !msg.isLoading);
        const errorMsg = { 
          id: idCounter++, 
          from: 'bot', 
          text: `죄송합니다. 오류가 발생했습니다: ${error.message}\n\n백엔드 서버가 실행 중인지 확인해주세요. (http://localhost:5000)`, 
          ts: Date.now() 
        };
        updated[sessionIdx].messages = [...filteredMessages, errorMsg];
        return updated;
      });
    }
  }

  function handleSelectSession(idx) {
    setCurrentSessionIdx(idx);
    setPendingSession(null);
  }

  function handleNewChat() {
    setPendingSession(makeDefaultSession());
  }

  function handleLogout() {
    setUser(null);
    setShowLogin(false);
    setShowSignup(false);
  }

  // 로그인 화면이 우선적으로 보이도록 분기
  if (!user && showLogin) {
    return (
      <Login
        onLogin={email => {
          setUser(email);
          setShowLogin(false);
        }}
        onSignup={() => {
          setShowSignup(true);
          setShowLogin(false);
        }}
      />
    );
  }

  // 회원가입 화면 분기
  if (!user && showSignup) {
    return (
      <Signup
        onSignup={email => {
          setUser(email);
          setShowSignup(false);
        }}
      />
    );
  }

  // StudyPlan 페이지
  if (currentPage === 'studyplan') {
    return (
      <div className="app-bg">
        <Header title="교수용 수업계획서" />
        <div style={{ 
          paddingTop: '0px', 
          paddingBottom: '45px', 
          minHeight: '100vh',
          boxSizing: 'border-box' 
        }}>
          <StudyPlan />
        </div>
        <Footer 
          currentPage={currentPage}
          onPageChange={setCurrentPage}
        />
      </div>
    );
  }

  // 챗봇 UI
  return (
    <div className="app-bg">
      <div className="app-center-box" style={sidebarOpen ? { paddingRight: 360 } : {}}>
        <Header title="AI 챗봇" />
        <ChatWindow
          messages={pendingSession ? pendingSession.messages : sessions[currentSessionIdx]?.messages}
          onSend={handleSend}
          sidebarOpen={sidebarOpen}
        />
      </div>
      <Sidebar
        open={sidebarOpen}
        sessions={sessions}
        currentSessionIdx={currentSessionIdx}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />
      <RightToolbar
        onToggle={() => setSidebarOpen((s) => !s)}
        sidebarOpen={sidebarOpen}
        onNewChat={handleNewChat}
        onLoginClick={() => {
          setShowLogin(true);
          setShowSignup(false);
        }}
        onLogoutClick={handleLogout}
        isLoggedIn={!!user}
      />
      <Footer 
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />
    </div>
  );
}

export default App;
