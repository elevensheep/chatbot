// API 기본 설정
// 런타임 환경변수 우선 사용 (window._env_는 nginx 시작 시 주입됨)
const API_BASE_URL = (window._env_ && window._env_.REACT_APP_API_URL) 
  || process.env.REACT_APP_API_URL 
  || 'http://localhost:5000/api';

// API 호출 유틸리티 함수
export const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;

  console.log('apiCall 시작:', url);
  console.log('API_BASE_URL:', API_BASE_URL);

  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const config = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };

  console.log('fetch 호출 전, config:', config);

  try {
    console.log('fetch 호출 시작...');

    // 타임아웃 설정 (60초) - AI 답변 생성에 시간이 걸리므로 충분한 시간 확보
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.log('fetch 타임아웃 발생!');
      controller.abort();
    }, 60000);

    const response = await fetch(url, { ...config, signal: controller.signal });
    clearTimeout(timeoutId);

    console.log('fetch 응답 받음:', response);
    console.log('응답 상태:', response.status, response.statusText);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      // 백엔드 에러 응답 구조 처리
      // { detail: { error: { message: "..." } } } 또는 { detail: "..." } 또는 { message: "..." }
      let errorMessage = `HTTP error! status: ${response.status}`;

      if (errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (errorData.detail.error?.message) {
          errorMessage = errorData.detail.error.message;
        } else if (errorData.detail.message) {
          errorMessage = errorData.detail.message;
        }
      } else if (errorData.message) {
        errorMessage = errorData.message;
      }

      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    console.error('에러 타입:', error.name);
    console.error('에러 메시지:', error.message);
    console.error('에러 전체:', error);
    throw error;
  }
};

// 인증 관련 API
export const authAPI = {
  // 로그인
  login: async (email, password) => {
    return apiCall('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  // 회원가입
  signup: async (userData) => {
    return apiCall('/auth/signup', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  // 이메일 인증번호 발송
  sendVerificationEmail: async (email) => {
    return apiCall('/auth/send-verification-email', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  // 이메일 인증번호 확인
  verifyEmail: async (email, code) => {
    return apiCall('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ email, code }),
    });
  },

  // 이메일 토큰으로 검증 (이메일 내 링크 클릭 시 사용)
  verifyWithToken: async (token) => {
    return apiCall('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    });
  },

  // 로그아웃
  logout: async () => {
    return apiCall('/auth/logout', {
      method: 'POST',
    });
  },

  // 토큰 검증
  verifyToken: async (token) => {
    return apiCall('/auth/verify', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });
  },

  // 비밀번호 재설정 요청
  requestPasswordReset: async (email) => {
    return apiCall('/auth/password-reset/request', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  // 비밀번호 재설정 확인
  confirmPasswordReset: async (email, token, newPassword) => {
    return apiCall('/auth/password-reset/confirm', {
      method: 'POST',
      body: JSON.stringify({
        email,
        token,
        new_password: newPassword
      }),
    });
  },
};

// 대화방 목록 조회
export const getConversations = async () => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  // 백엔드 응답 구조: { success: true, data: { conversations: [...], total: N } }
  return response.data?.conversations || response.conversations || [];
};

// 새 대화방 생성
export const createConversation = async (title) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });
  // 백엔드 응답 구조: { success: true, data: {...} }
  return response.data || response;
};

// 특정 대화방 메시지 조회
export const getMessages = async (conversationId) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall(`/conversations/${conversationId}/messages`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  // 백엔드 응답 구조: { success: true, data: { messages: [...], total: N } }
  return response.data || response;
};

// 메시지 전송
export const sendMessage = async (conversationId, query, k = 3, includeSources = true) => {
  const token = localStorage.getItem('access_token');
  const response = await apiCall('/conversations/chat', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ conversation_id: conversationId, query, k, include_sources: includeSources }),
  });
  // 백엔드 응답 구조: { success: true, data: {...} }
  return response.data || response;
};

export { API_BASE_URL };