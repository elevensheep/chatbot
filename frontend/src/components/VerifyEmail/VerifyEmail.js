import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { authAPI } from '../../utils/api';
import './VerifyEmail.css';

export default function VerifyEmail() {
  const location = useLocation();
  const navigate = useNavigate();
  const [status, setStatus] = useState('처리중...');
  const [error, setError] = useState('');

  useEffect(() => {
    async function verify() {
      const params = new URLSearchParams(location.search);
      const token = params.get('token');

      if (!token) {
        setError('유효한 검증 토큰이 없습니다.');
        setStatus('실패');
        return;
      }

      try {
        const res = await authAPI.verifyWithToken(token);
        // backend returns { success: True, data: { email: 'user@bu.ac.kr', verified: True } }
        const verifiedEmail = res?.data?.email || res?.email;
        if (verifiedEmail) {
          // Redirect back to signup with verifiedEmail param
          const encoded = encodeURIComponent(verifiedEmail);
          navigate(`/signup?verifiedEmail=${encoded}`);
        } else {
          setError('이메일을 확인할 수 없습니다.');
          setStatus('실패');
        }
      } catch (err) {
        console.error('verify token failed', err);
        setError(err.message || '서버 오류로 인증에 실패했습니다.');
        setStatus('실패');
      }
    }

    verify();
  }, [location.search, navigate]);

  return (
    <div className="verify-page">
      <div className="verify-box">
        <h2>이메일 인증 중</h2>
        <p>{status}</p>
        {error && <p className="verify-error">{error}</p>}
      </div>
    </div>
  );
}
