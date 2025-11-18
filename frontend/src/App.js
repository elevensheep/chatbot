import React, { useState, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import './App.css';
import Header from './components/Header/Header';
import ChatWindow from './components/ChatWindow/ChatWindow';
import Login from './components/Login/Login';
import Signup from './components/Signup/Signup';
import VerifyEmail from './components/VerifyEmail/VerifyEmail';
import Sidebar from './components/Sidebar/Sidebar';
import RightToolbar from './components/RightToolbar/RightToolbar';
import Footer from './components/Footer/Footer';
import StudyPlan from './components/StudyPlan/StudyPlan';
import { LoadingOverlay } from './components/LoadingSpinner/LoadingSpinner';


// 고유한 메시지 ID 생성 함수
function generateMessageId() {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function makeDefaultSession() {
  return {
    id: Date.now(),
    messages: [
      { id: generateMessageId(), from: 'bot', text: '안녕하세요! 무엇을 도와드릴까요?', ts: Date.now() },
    ],
    created: Date.now(),
  };
}

function MainApp() {
  const navigate = useNavigate();
  // 상태 관리
  const [sessions, setSessions] = useState([makeDefaultSession()]);
  const [currentSessionIdx, setCurrentSessionIdx] = useState(0);
  const currentSessionIdxRef = useRef(currentSessionIdx);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pendingSession, setPendingSession] = useState(null);
  const [user, setUser] = useState(null);
  const [currentPage, setCurrentPage] = useState('chat'); // 페이지 상태 추가
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태 추가
  const [loadingMessage, setLoadingMessage] = useState(''); // 로딩 메시지
  const isRequestingRef = useRef(false); // 중복 요청 방지 플래그

  React.useEffect(() => {
    currentSessionIdxRef.current = currentSessionIdx;
  }, [currentSessionIdx]);

  async function callChatAPI(userMessage) {
    // 이미 요청 중이면 중복 요청 방지
    if (isRequestingRef.current) {
      console.warn('이미 요청이 진행 중입니다. 중복 요청을 무시합니다.');
      return '요청이 이미 진행 중입니다. 잠시만 기다려주세요.';
    }
    
    isRequestingRef.current = true;
    setIsLoading(true);
    setLoadingMessage('질문을 분석하고 있습니다...');
    
    try {
      // 단계별 로딩 메시지
      setTimeout(() => {
        setLoadingMessage('관련 정보를 검색하고 있습니다...');
      }, 1000);
      
      setTimeout(() => {
        setLoadingMessage('AI가 답변을 생성하고 있습니다...');
      }, 2000);
      
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          k: 3,
          include_sources: true
        }),
        // 타임아웃 설정 (모델 로딩 시간 고려)
        signal: AbortSignal.timeout(300000) // 5분 타임아웃
      });

      if (!response.ok) {
        throw new Error(`API 호출 실패: ${response.status}`);
      }

      const data = await response.json();
      return data.answer || '답변을 생성할 수 없습니다.';
    } catch (error) {
      console.error('백엔드 API 호출 오류:', error);
      
      // 타임아웃 에러 처리
      if (error.name === 'TimeoutError' || error.name === 'AbortError') {
        return '요청 시간이 초과되었습니다. 모델이 로딩 중일 수 있습니다. 잠시 후 다시 시도해주세요.';
      }
      
      // 네트워크 에러 처리
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        return '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.';
      }
      
      return `죄송합니다. 서버와 통신 중 오류가 발생했습니다: ${error.message}`;
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
      isRequestingRef.current = false; // 요청 완료 후 플래그 해제
    }
  }

  function handleSend(text) {
    // 줄바꿈을 공백으로 변환 (이미 MessageInput에서 처리되지만 이중 보장)
    const cleanText = text.replace(/\n/g, ' ').trim();
    
    if (pendingSession) {
      const newSession = {
        ...pendingSession,
        messages: [
          ...pendingSession.messages,
          { id: generateMessageId(), from: 'user', text: cleanText, ts: Date.now() }
        ]
      };
      const newSessionIdx = sessions.length;
      
      setSessions(prev => [...prev, newSession]);
      setCurrentSessionIdx(newSessionIdx);
      setPendingSession(null);
      
      // 백엔드 API 호출 (setSessions 밖에서 호출)
      callChatAPI(cleanText).then(botResponse => {
        setSessions(prev2 => {
          const updated2 = [...prev2];
          const botMsg = { id: generateMessageId(), from: 'bot', text: botResponse, ts: Date.now() };
          if (updated2[newSessionIdx]) {
            updated2[newSessionIdx].messages = [...updated2[newSessionIdx].messages, botMsg];
          }
          return updated2;
        });
      });
      return;
    }
    
    const idx = currentSessionIdxRef.current;
    const loadingMsgId = generateMessageId();
    
    // 사용자 메시지 추가
    setSessions(prev => {
      return prev.map((session, i) =>
        i === idx
          ? { 
              ...session, 
              messages: [
                ...session.messages, 
                { id: generateMessageId(), from: 'user', text: cleanText, ts: Date.now() },
                { id: loadingMsgId, from: 'bot', text: '답변을 생성하는 중...', ts: Date.now(), isLoading: true }
              ] 
            }
          : session
      );
    });
    
    // 백엔드 API 호출 (setSessions 밖에서 호출)
    callChatAPI(cleanText).then(botResponse => {
      setSessions(prev2 => {
        const updated2 = prev2.map((session, i) => {
          if (i === idx) {
            // 로딩 메시지 제거하고 실제 응답 추가
            const filtered = session.messages.filter(m => m.id !== loadingMsgId);
            return { ...session, messages: [...filtered, { id: generateMessageId(), from: 'bot', text: botResponse, ts: Date.now() }] };
          }
          return session;
        });
        return updated2;
      });
    });
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
    navigate('/');
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
        
        {/* 로딩 오버레이 */}
        {isLoading && (
          <LoadingOverlay message={loadingMessage} />
        )}
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
          navigate('/login');
        }}
        onLogoutClick={handleLogout}
        isLoggedIn={!!user}
      />
      <Footer 
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />
      
      {/* 로딩 오버레이 */}
      {isLoading && (
        <LoadingOverlay message={loadingMessage} />
      )}
    </div>
  );
}

// 독립적인 로그인 컴포넌트
function LoginPage() {
  const navigate = useNavigate();
  
  return (
    <Login
      onLogin={email => {
        // 로그인 성공 후 메인 페이지로 이동
        navigate('/');
      }}
      onSignup={() => {
        navigate('/signup');
      }}
      onBack={() => {
        navigate('/');
      }}
    />
  );
}

// 독립적인 회원가입 컴포넌트
function SignupPage() {
  const navigate = useNavigate();
  
  return (
    <Signup
      onSignup={email => {
        // 회원가입 성공 후 메인 페이지로 이동
        navigate('/');
      }}
      onBack={() => {
        navigate('/login');
      }}
    />
  );
}

// 메인 App 컴포넌트 (라우터 설정)
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainApp />} />
        <Route path="/login" element={<LoginPage />} />
  <Route path="/signup" element={<SignupPage />} />
  <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
