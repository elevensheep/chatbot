import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import './App.css';
import Header from './components/Header/Header';
import ChatWindow from './components/ChatWindow/ChatWindow';
import Login from './components/Login/Login';
import FindPassword from './components/Login/FindPassword';
import Signup from './components/Signup/Signup';
import VerifyEmail from './components/VerifyEmail/VerifyEmail';
import Sidebar from './components/Sidebar/Sidebar';
import RightToolbar from './components/RightToolbar/RightToolbar';
import Footer from './components/Footer/Footer';
import StudyPlan from './components/StudyPlan/StudyPlan';
import { LoadingOverlay } from './components/LoadingSpinner/LoadingSpinner';
import { getConversations, createConversation, getMessages, sendMessage as sendMessageAPI } from './utils/api';


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
  // 채팅방 목록 상태
  const [conversations, setConversations] = useState([]);
  // 현재 선택된 채팅방 ID
  const [currentConversationId, setCurrentConversationId] = useState(null);
  // 상태 관리 - 채팅방별 메시지 저장
  const [sessions, setSessions] = useState(() => {
    return [];
  });
  const [currentSessionIdx, setCurrentSessionIdx] = useState(0);
  const currentSessionIdxRef = useRef(currentSessionIdx);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pendingSession, setPendingSession] = useState(null);
  const [user, setUser] = useState(() => {
    // 초기 렌더링 시 localStorage에서 사용자 정보 복원
    const savedEmail = localStorage.getItem('userEmail');
    return savedEmail || null;
  });
  const [currentPage, setCurrentPage] = useState('chat');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [showFindPassword, setShowFindPassword] = useState(false);

  // 로그인 시 채팅방 목록 로드
  useEffect(() => {
    const isLoggedIn = (!!localStorage.getItem('authToken') && !!localStorage.getItem('access_token')) || !!user;
    if (isLoggedIn) {
      loadConversations();
    }
  }, [user]);

  // 채팅방 목록 로드 함수
  const loadConversations = async () => {
    try {
      const convs = await getConversations();
      setConversations(convs || []);
      
      // 첫 번째 채팅방이 있으면 자동으로 로드
      if (convs && convs.length > 0 && !currentConversationId) {
        await loadConversationMessages(convs[0].id);
      }
    } catch (error) {
      console.error('채팅방 목록 로드 실패:', error);
    }
  };

  // 특정 채팅방의 메시지 로드
  const loadConversationMessages = async (conversationId) => {
    try {
      setIsLoading(true);
      setLoadingMessage('메시지를 불러오는 중...');
      
      const messagesData = await getMessages(conversationId);
      const messages = messagesData.messages || messagesData || [];
      
      // 백엔드 메시지 형식을 프론트엔드 형식으로 변환
      const formattedMessages = messages.map((msg, index) => ({
        id: msg.id || `msg_${index}`,
        from: msg.role === 'user' ? 'user' : 'bot',
        text: msg.content || '',
        sources: msg.sources || [],
        ts: new Date(msg.created_at).getTime() || Date.now()
      }));
      
      // 환영 메시지가 없으면 추가
      if (formattedMessages.length === 0) {
        formattedMessages.push({
          id: generateMessageId(),
          from: 'bot',
          text: '안녕하세요! 무엇을 도와드릴까요?',
          ts: Date.now()
        });
      }
      
      // 세션에 추가 또는 업데이트
      setSessions(prev => {
        const existingIdx = prev.findIndex(s => s.conversationId === conversationId);
        const newSession = {
          id: conversationId,
          conversationId: conversationId,
          messages: formattedMessages,
          created: Date.now()
        };
        
        let newSessions;
        if (existingIdx >= 0) {
          newSessions = [...prev];
          newSessions[existingIdx] = newSession;
        } else {
          newSessions = [newSession, ...prev];
        }
        
        // currentSessionIdx 업데이트
        const sessionIdx = newSessions.findIndex(s => s.conversationId === conversationId);
        setCurrentSessionIdx(sessionIdx >= 0 ? sessionIdx : 0);
        
        return newSessions;
      });
      
      setCurrentConversationId(conversationId);
      
    } catch (error) {
      console.error('메시지 로드 실패:', error);
      alert('메시지를 불러오는데 실패했습니다: ' + error.message);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  // 로그인 상태 확인 (localStorage와 user 상태 모두 체크)
  const isLoggedIn = (!!localStorage.getItem('authToken') && !!localStorage.getItem('access_token')) || !!user;
  if (!isLoggedIn) {
    if (showFindPassword) {
      return (
        <FindPassword onBack={() => setShowFindPassword(false)} />
      );
    }
    return (
      <Login
        onLogin={email => {
          setUser(email);
          // navigate는 필요 없음 - setUser가 리렌더링을 트리거함
        }}
        onSignup={() => {
          navigate('/signup');
        }}
        onBack={() => {
          navigate('/');
        }}
        onFindPassword={() => setShowFindPassword(true)}
      />
    );
  }

  // ...기존 챗봇 UI 렌더링 로직 (생략, 기존 코드 유지)...
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
        {isLoading && (
          <LoadingOverlay message={loadingMessage} />
        )}
      </div>
    );
  }

  // 새 채팅 생성 핸들러
  const handleNewChat = async () => {
    try {
      setIsLoading(true);
      setLoadingMessage('새 채팅방을 생성하는 중...');
      
      // 새 채팅방 생성
      const newConv = await createConversation('새 대화');
      const conversationId = newConv.conversation_id || newConv.id;
      
      // 채팅방 목록 업데이트
      await loadConversations();
      
      // 새 채팅방의 메시지 로드 (환영 메시지 포함)
      await loadConversationMessages(conversationId);
      
    } catch (error) {
      console.error('새 채팅 생성 실패:', error);
      alert('새 채팅방 생성에 실패했습니다: ' + error.message);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  // 메시지 전송 핸들러
  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    // 현재 채팅방이 없으면 새로 생성
    let conversationId = currentConversationId;
    if (!conversationId) {
      try {
        const newConv = await createConversation('새 대화');
        conversationId = newConv.conversation_id || newConv.id;
        setCurrentConversationId(conversationId);
        await loadConversations();
      } catch (error) {
        console.error('채팅방 생성 실패:', error);
        alert('채팅방 생성에 실패했습니다: ' + error.message);
        return;
      }
    }

    setIsLoading(true);
    setLoadingMessage('답변을 생성하고 있습니다...');

    try {
      // 로컬에 사용자 메시지 추가
      const userMessage = {
        id: generateMessageId(),
        from: 'user',
        text: text.trim(),
        ts: Date.now()
      };

      const currentSession = sessions.find(s => s.conversationId === conversationId);
      const currentMessages = currentSession?.messages || [];
      const updatedMessages = [...currentMessages, userMessage];

      // 세션 업데이트
      setSessions(prev => {
        const existingIdx = prev.findIndex(s => s.conversationId === conversationId);
        const updatedSession = {
          id: conversationId,
          conversationId: conversationId,
          messages: updatedMessages,
          created: currentSession?.created || Date.now()
        };
        
        if (existingIdx >= 0) {
          const newSessions = [...prev];
          newSessions[existingIdx] = updatedSession;
          return newSessions;
        } else {
          return [updatedSession, ...prev];
        }
      });

      // 백엔드 API 호출
      const response = await sendMessageAPI(conversationId, text.trim(), 3, true);

      // 봇 응답 추가
      const botMessage = {
        id: generateMessageId(),
        from: 'bot',
        text: response.answer || response.data?.answer || '응답을 받지 못했습니다.',
        sources: response.sources || response.data?.sources || [],
        ts: Date.now()
      };

      // 세션에 봇 메시지 추가
      setSessions(prev => {
        const existingIdx = prev.findIndex(s => s.conversationId === conversationId);
        if (existingIdx >= 0) {
          const newSessions = [...prev];
          newSessions[existingIdx] = {
            ...newSessions[existingIdx],
            messages: [...updatedMessages, botMessage]
          };
          return newSessions;
        }
        return prev;
      });

      // 채팅방 목록 새로고침 (updated_at 업데이트 반영)
      await loadConversations();

    } catch (error) {
      console.error('메시지 전송 실패:', error);
      alert('메시지 전송에 실패했습니다: ' + error.message);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  // 챗봇 UI
  return (
    <div className="app-bg">
      <div className="app-center-box" style={sidebarOpen ? { paddingRight: 360 } : {}}>
        <Header title="수업 플래너 챗봇" />
        <ChatWindow
          messages={pendingSession ? pendingSession.messages : (sessions.find(s => s.conversationId === currentConversationId)?.messages || [])}
          onSend={handleSendMessage}
          sidebarOpen={sidebarOpen}
        />
      </div>
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={loadConversationMessages}
        onNewChat={handleNewChat}
        onLogout={() => {
          setUser(null);
          setConversations([]);
          setCurrentConversationId(null);
          setSessions([]);
          localStorage.removeItem('authToken');
          localStorage.removeItem('userEmail');
          localStorage.removeItem('access_token');
          navigate('/login');
        }}
      />
      <RightToolbar
        onToggle={() => setSidebarOpen((s) => !s)}
        sidebarOpen={sidebarOpen}
        onNewChat={handleNewChat}
        onLoginClick={() => {
          navigate('/login');
        }}
        onLogoutClick={() => {
          setUser(null);
          localStorage.removeItem('authToken');
          localStorage.removeItem('userEmail');
          localStorage.removeItem('access_token');
          navigate('/login');
        }}
        isLoggedIn={!!user}
      />
      <Footer
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />
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
