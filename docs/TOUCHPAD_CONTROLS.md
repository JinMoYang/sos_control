# RayNeo X3 Pro 터치패드 조작

SoS 앱은 RayNeo 다리 터치패드를 화면 좌표가 아닌 **포커스 컨트롤러**로 처리한다. 현재 선택된 항목은 약간 확대되어 표시되고, 긴 설정 화면에서는 해당 항목이 보이도록 자동 스크롤된다.

| 터치 동작 | 앱 동작 |
|---|---|
| 앞쪽 또는 위쪽으로 스와이프 | 다음으로 조작할 수 있는 항목으로 이동 |
| 뒤쪽 또는 아래쪽으로 스와이프 | 이전 항목으로 이동 |
| 짧게 탭 | 현재 항목 선택 또는 버튼 실행 |
| 길게 누르기 | 측정 중이 아니면 현재 선택한 Method 시작, 측정 중이면 정지 |
| Back | 측정 중이면 정지만 수행하고 앱에 머무름. 정지 상태이면 앱 화면을 나감 |

## 항목별 탭 동작

- Method, boost, AE strategy, Proposed period, fallback: 해당 라디오 옵션을 선택한다.
- Fixed cell: 해당 ISO/노출 후보를 선택한다.
- Confidence와 Network spinner: 팝업을 열지 않고 탭할 때마다 다음 값으로 순환한다.
- Offload: 켜기/끄기를 전환한다. 서버 URL은 글래스 키보드가 뜨는 일을 피하기 위해 터치패드 포커스에서 제외했다.
- Start/Stop/Verify/VeriProbe/Bench/IsoDiag: 해당 버튼을 즉시 실행한다.

앱 시작 시 기본 포커스는 **Start**이며 기본 Method는 **Proposed**다. 따라서 별도 설정이 필요하지 않으면 짧게 한 번 탭하거나 어디서든 길게 눌러 Proposed 측정을 시작할 수 있다.

키보드 또는 리모컨을 연결한 경우 방향키·Tab·Page Up/Down으로 이동하고 Enter·Space·DPAD Center로 실행할 수 있다.
