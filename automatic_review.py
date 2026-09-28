"""Existing editor launcher with cloud proposals and an explicit local mode."""
import sys,os,json,traceback,hashlib
from pathlib import Path
from app_paths import ROOT,add_dependencies
add_dependencies()
from PySide6.QtCore import QThread,Signal,QTimer,Qt,QPropertyAnimation,QEasingCurve,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QComboBox,QFrame,QGraphicsOpacityEffect,QSizeGrip
from cut_review import configure_app
from review_core import save_json
from automatic_cut import probe,plan,send,check_resolve,Cancelled
from retention_dial import RetentionDial
from model_assets import status as model_status,ensure_models,SetupCancelled

def log_worker_error(filename,detail):
    secret=os.environ.get('OPENAI_API_KEY','')
    if secret:detail=detail.replace(secret,'[REDACTED]')
    try:
        (ROOT/'analysis').mkdir(parents=True,exist_ok=True)
        (ROOT/'analysis'/filename).write_text(detail,encoding='utf-8')
    except OSError:
        # A full/read-only disk must not suppress the visible error signal.
        if sys.stderr:print(detail,file=sys.stderr,flush=True)

class ModelSetupWorker(QThread):
    progress=Signal(str);ready=Signal(dict);failed=Signal(str)
    def __init__(self,groups=None):
        super().__init__();self.groups=groups
    def run(self):
        try:self.ready.emit(ensure_models(self.progress.emit,self.isInterruptionRequested,groups=self.groups))
        except SetupCancelled as error:self.failed.emit(str(error))
        except Exception:
            detail=traceback.format_exc();log_worker_error('model-setup-error.log',detail)
            self.failed.emit(detail.splitlines()[-1])

class SoftLabel(QLabel):
    """Interruptible fades: frequent worker messages never queue animations."""
    def __init__(self,*args):
        super().__init__(*args)
        effect=QGraphicsOpacityEffect(self);effect.setOpacity(1);self.setGraphicsEffect(effect)
        self.fade=QPropertyAnimation(effect,b'opacity',self)
        self.fade.setDuration(260);self.fade.setEasingCurve(QEasingCurve.OutCubic)
    def setText(self,text):
        if text==self.text():return
        super().setText(text)
        if self.isVisible() and os.environ.get('RETENTION_REDUCED_MOTION')!='1':
            self.fade.stop();self.fade.setStartValue(.4);self.fade.setEndValue(1);self.fade.start()

class SoftButton(QPushButton):
    def __init__(self,*args):
        super().__init__(*args)
        effect=QGraphicsOpacityEffect(self);effect.setOpacity(.92);self.setGraphicsEffect(effect)
        self.hover=QPropertyAnimation(effect,b'opacity',self)
        self.hover.setDuration(180);self.hover.setEasingCurve(QEasingCurve.OutCubic)
    def transition(self,value):
        self.hover.stop()
        if os.environ.get('RETENTION_REDUCED_MOTION')=='1':self.graphicsEffect().setOpacity(value);return
        self.hover.setStartValue(self.graphicsEffect().opacity());self.hover.setEndValue(value);self.hover.start()
    def enterEvent(self,event):self.transition(1);super().enterEvent(event)
    def leaveEvent(self,event):self.transition(.92);super().leaveEvent(event)

class WindowHeader(QWidget):
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            handle=self.window().windowHandle()
            if handle and handle.startSystemMove():event.accept();return
            self.drag_offset=event.globalPosition().toPoint()-self.window().pos()
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if event.buttons() & Qt.LeftButton and hasattr(self,'drag_offset'):
            self.window().move(event.globalPosition().toPoint()-self.drag_offset)
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if hasattr(self,'drag_offset'):del self.drag_offset
        super().mouseReleaseEvent(event)

class Worker(QThread):
    progress=Signal(str);ready=Signal(dict);failed=Signal(str)
    def __init__(self,source,mic,intensity='balanced',backend='local'):
        super().__init__();self.source=source;self.mic=mic;self.intensity=intensity;self.backend=backend
    def run(self):
        try:
            if self.backend=='openai':
                from ai_pipeline import plan_ai
                result=plan_ai(self.source,self.mic,self.progress.emit,self.isInterruptionRequested,intensity=self.intensity)
                if self.isInterruptionRequested():raise Cancelled()
                self.ready.emit(result)
                return
            result=plan(self.source,self.mic,self.progress.emit,self.isInterruptionRequested,intensity=self.intensity)
            if self.isInterruptionRequested():raise Cancelled()
            self.ready.emit(send(result,self.progress.emit))
        except Cancelled:self.failed.emit('Stopped before sending the cut to Resolve.')
        except Exception:
            detail=traceback.format_exc();log_worker_error('automatic-error.log',detail)
            secret=os.environ.get('OPENAI_API_KEY','')
            if secret:detail=detail.replace(secret,'[REDACTED]')
            self.failed.emit(detail.splitlines()[-1])

class AutomaticWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('clipemo.com');self.resize(480,740);self.setMinimumSize(440,720)
        self.setWindowFlag(Qt.FramelessWindowHint,True)
        self.setAttribute(Qt.WA_TranslucentBackground,True)
        self.setStyleSheet('''
            QMainWindow { background: #11131c; }
            QWidget#canvas { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #1c2033,stop:0.55 #141722,stop:1 #11131c); }
            QWidget { font-family: 'SF Pro Display'; font-size: 13px; color: #f1f2f7; }
            QLabel { background: transparent; border: none; }
            QLabel#title { font-size: 23px; font-weight: 600; letter-spacing: -0.5px; }
            QLabel#muted, QLabel#caption { color: #a6adc3; }
            QLabel#eyebrow { color: #74747c; font-size: 11px; font-weight: 600; letter-spacing: 2px; }
            QLabel#section { font-size: 13px; font-weight: 600; color: #cbd0df; }
            QLabel#mark { background: #2d3551; color: #dce3ff; border: 1px solid #454f72; border-radius: 12px; font-size: 22px; font-weight: 600; }
            QLabel#badge { color: #52615c; background: #e9eeeb; border-radius: 12px; padding: 6px 12px; font-size: 11px; }
            QFrame#card { background: #202434; border: 1px solid #33394e; border-radius: 16px; }
            QPushButton { background: #2a3043; border: 1px solid #41495f; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: #2a3043; }
            QPushButton:pressed { background: #3b435a; }
            QPushButton:focus { border: 2px solid #a8b9ff; }
            QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #536dd9,stop:1 #7277d9); color: white; border: 1px solid #8493ed; border-radius: 10px; font-size: 14px; }
            QPushButton#primary:pressed { background: #485abd; }
            QPushButton#primary:focus { border: 2px solid #d0d9ff; }
            QPushButton:disabled, QPushButton#primary:disabled { background: #272c3d; color: #929bb2; border: 1px solid #353c51; }
            QComboBox { background: #191d2b; border: 1px solid #41495f; border-radius: 8px; padding: 8px; }
            QComboBox QAbstractItemView { background: #202434; color: #f1f2f7; selection-background-color: #465584; }
            QProgressBar { border: none; background: #292e40; border-radius: 2px; max-height: 4px; }
            QProgressBar::chunk { background: #8a9aee; border-radius: 2px; }
        ''')
        self.source=None;self.worker=None;self.result=None;self.setup_worker=None;self.review_dialog=None
        self.models_ready=False;self.close_after_setup=False
        self.setStyleSheet(self.styleSheet()+'''
            QWidget#canvas { background: qradialgradient(cx:0.16,cy:0.06,radius:1.15,fx:0.16,fy:0.06,stop:0 #43405b,stop:0.32 #2b2b42,stop:0.72 #202233,stop:1 #171a28); }
            QLabel#title { font-size: 16px; font-weight: 600; letter-spacing: 0px; }
            QFrame#dock { background: qlineargradient(x1:0,y1:0,x2:0.8,y2:1,stop:0 #3b384f,stop:0.45 #302e44,stop:1 #26283b); border: 1px solid #48445b; border-bottom-color: #292b3c; border-radius: 26px; }
            QPushButton { background: #39364e; border: 1px solid #504b65; border-radius: 16px; padding: 8px 16px; }
            QPushButton:hover { background: #39364e; }
            QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:0.8,y2:1,stop:0 #a282ff,stop:0.16 #9066fb,stop:0.55 #7a50ee,stop:1 #6040d1); border: 1px solid #a98afa; border-bottom-color: #6950bc; border-radius: 26px; font-size: 16px; }
            QPushButton#primary:pressed { background: #6542d9; }
            QPushButton:focus, QPushButton#primary:focus { border: 1px solid #c8b4ff; }
        ''')
        self.setStyleSheet(self.styleSheet()+'''
            QWidget#canvas { background: qlineargradient(x1:0,y1:0,x2:0.7,y2:1,stop:0 #282d35,stop:0.45 #1c2128,stop:1 #14191f); }
            QWidget { color: #eef1f5; }
            QLabel#title { font-family: 'SF Pro Display'; font-size: 19px; font-weight: 700; }
            QLabel#caption { color: #a8b1bd; }
            QLabel#section { color: #e0e5eb; font-size: 13px; }
            QFrame#dock { background: qlineargradient(x1:0,y1:0,x2:0.7,y2:1,stop:0 #2a3038,stop:1 #20262e); border: 1px solid #3c444f; border-bottom-color: #262e37; border-radius: 23px; }
            QPushButton { color: #dce2e9; background: #343c46; border: none; }
            QPushButton:hover { background: #343c46; }
            QPushButton#primary { color: #192331; background: qlineargradient(x1:0,y1:0,x2:0.3,y2:1,stop:0 #e0e9f5,stop:1 #b7cbe5); border: 1px solid #e4edf9; border-bottom-color: #98acc4; border-radius: 18px; font-size: 15px; }
            QPushButton#primary:pressed { background: #a9bfd9; }
            QPushButton:focus, QPushButton#primary:focus { border: 1px solid #f1f5fc; }
            QPushButton#primary:disabled { color: #aab4c0; background: #333c47; border: 1px solid #48535f; }
        ''')
        self.setStyleSheet(self.styleSheet()+'''
            QMainWindow { background: transparent; }
            QWidget#canvas { border: 1px solid #424a55; border-bottom-color: #252e39; border-radius: 22px; }
            QFrame#dock { background: qlineargradient(x1:0,y1:0,x2:0.6,y2:1,stop:0 #323a44,stop:0.12 #2a323b,stop:1 #20262e); border: 1px solid #46515e; border-right-color: #303a46; border-bottom-color: #171e26; border-radius: 23px; }
            QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f0f5fd,stop:0.09 #dce7f5,stop:0.8 #b7cbe5,stop:1 #a4bad5); border: 1px solid #e5eef9; border-bottom: 2px solid #8198b3; }
            QPushButton#primary:pressed { background: #afc3dc; border: 1px solid #8fa5c0; }
            QPushButton#primary:disabled { background: #333c47; border: 1px solid #48535f; }
            QSizeGrip { background: transparent; }
            QPushButton#secondary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #414c59,stop:1 #303943); border: 1px solid #4c5867; border-bottom-color: #252e38; }
            QPushButton#secondary:pressed { background: #29323c; border-color: #394452; }
            QPushButton#secondary:disabled { background: #2b333d; color: #8f9caa; border-color: #36414c; }
        ''')
        base=QWidget();base.setObjectName('canvas');self.setCentralWidget(base);layout=QVBoxLayout(base);layout.setContentsMargins(22,24,22,18);layout.setSpacing(10)
        header=WindowHeader();header_layout=QHBoxLayout(header);header_layout.setContentsMargins(0,0,0,0);header_layout.setSpacing(5)
        title=QLabel('clipemo.com');title.setObjectName('title');title.setAttribute(Qt.WA_TransparentForMouseEvents);header_layout.addWidget(title);header_layout.addStretch()
        self.minimize=QPushButton('−');self.minimize.setAccessibleName('Minimize window');self.minimize.setToolTip('Minimize');self.minimize.clicked.connect(self.showMinimized)
        self.close_button=QPushButton('×');self.close_button.setAccessibleName('Close window');self.close_button.setToolTip('Close');self.close_button.clicked.connect(self.close)
        for button in (self.minimize,self.close_button):
            button.setFixedSize(32,30);button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet('QPushButton { background:transparent; border:none; padding:0; font-size:22px; color:#abb5c1; } QPushButton:hover { background:transparent; color:white; } QPushButton:focus { border:none; color:#b9dcff; } QPushButton:pressed { color:#8296ab; }')
            header_layout.addWidget(button)
        layout.addWidget(header)
        self.dial=RetentionDial();self.dial.setMinimumHeight(180);layout.addWidget(self.dial,1)
        dock=QFrame();dock.setObjectName('dock');source_card=QVBoxLayout(dock);source_card.setContentsMargins(24,22,24,20);source_card.setSpacing(12);layout.addWidget(dock)
        row=QHBoxLayout();heading=QLabel('Source recording');heading.setObjectName('caption');row.addWidget(heading);row.addStretch()
        self.choose=SoftButton('Choose file');self.choose.setObjectName('secondary');self.choose.setCursor(Qt.PointingHandCursor);self.choose.clicked.connect(self.pick);row.addWidget(self.choose);source_card.addLayout(row)
        self.file=SoftLabel('Choose a recording to get started.');self.file.setWordWrap(True);self.file.setTextFormat(Qt.PlainText);self.file.setMinimumHeight(42);source_card.addWidget(self.file)
        self.voice_label=QLabel('Commentary track');self.voice_label.setObjectName('caption');source_card.addWidget(self.voice_label)
        self.voice=QComboBox();self.voice.setAccessibleName('Commentary audio track');source_card.addWidget(self.voice)
        backend_row=QHBoxLayout();backend_row.addWidget(QLabel('Decisions'))
        self.backend=QComboBox();self.backend.setAccessibleName('Decision backend')
        self.backend.addItem('OpenAI · review before export','openai')
        self.backend.addItem('Local · original automatic editor','local')
        self.default_backend='openai' if os.environ.get('OPENAI_API_KEY','').strip() else 'local'
        self.backend.setCurrentIndex(self.backend.findData(self.default_backend))
        self.backend.setToolTip('OpenAI sends commentary audio and transcript context to the API. Local uses only installed models.')
        backend_row.addWidget(self.backend,1);source_card.addLayout(backend_row)
        pacing=QHBoxLayout();pacing.addWidget(QLabel('Pacing'))
        self.intensity=QComboBox();self.intensity.setAccessibleName('Editing intensity')
        for label,value in [('Natural','natural'),('Balanced','balanced'),('Aggressive','aggressive')]:self.intensity.addItem(label,value)
        self.intensity.setCurrentIndex(1)
        self.intensity.setToolTip('Controls pause lengths and speech handles. Meaning and uncertainty protections apply in every mode.')
        pacing.addWidget(self.intensity,1);source_card.addLayout(pacing)
        self.create=SoftButton('Create automatic cut');self.create.setObjectName('primary');self.create.setMinimumHeight(54);self.create.clicked.connect(self.begin);self.create.setEnabled(False)
        self.create.setCursor(Qt.PointingHandCursor)
        progress_card=QVBoxLayout();progress_card.setContentsMargins(0,0,0,0);progress_card.setSpacing(3);source_card.addLayout(progress_card)
        self.stage=SoftLabel('Ready when you are');self.stage.setObjectName('section');progress_card.addWidget(self.stage)
        self.status=SoftLabel('Open a project in Resolve to begin.');self.status.setObjectName('caption');self.status.setTextFormat(Qt.PlainText);self.status.setWordWrap(True);self.status.setMinimumHeight(44);progress_card.addWidget(self.status)
        self.setup_button=SoftButton('Set up local AI');self.setup_button.setObjectName('secondary')
        self.setup_button.clicked.connect(self.start_model_setup);source_card.addWidget(self.setup_button)
        source_card.addWidget(self.create)
        footer=QHBoxLayout();footer.addSpacing(16)
        self.details=QLabel('DaVinci Resolve');self.details.setAlignment(Qt.AlignCenter);self.details.setStyleSheet('color: #9ca7b4; font-size: 11px;');footer.addWidget(self.details,1)
        self.details.linkActivated.connect(self.open_export_folder)
        grip=QSizeGrip(self);grip.setFixedSize(16,16);footer.addWidget(grip);layout.addLayout(footer)
        self.voice.hide();self.voice_label.hide()
        try:
            preferences=json.loads((ROOT/'analysis/simple-preferences.json').read_text())
            preferred=preferences.get('intensity','balanced')
            index=self.intensity.findData('aggressive' if preferred=='tight' else preferred)
            if index>=0:self.intensity.setCurrentIndex(index)
            preferred_backend=preferences.get('backend',self.default_backend)
            if preferred_backend=='openai' and self.default_backend=='local':
                preferred_backend='local'
            backend_index=self.backend.findData(preferred_backend)
            if backend_index>=0:self.backend.setCurrentIndex(backend_index)
            last=Path(preferences['source'])
            if last.is_file():self.load(last)
        except (OSError,ValueError,KeyError):pass
        self.backend.currentIndexChanged.connect(self.refresh_models)
        if '--source' in sys.argv:self.load(Path(sys.argv[sys.argv.index('--source')+1]))
        self.refresh_models()
    def refresh_models(self):
        cloud=self.backend.currentData()=='openai'
        readiness=model_status(groups=['speech']) if cloud else model_status()
        self.models_ready=readiness['ready'] and (not cloud or bool(os.environ.get('OPENAI_API_KEY','').strip()))
        self.setup_button.setVisible(not readiness['ready'])
        self.setup_button.setText('Set up speech timing' if cloud else 'Set up local AI')
        self.create.setEnabled(self.source is not None and self.models_ready)
        if not readiness['ready']:
            self.stage.setText('One-time setup needed')
            self.status.setText(f"Download {readiness['bytes_missing']/1e9:.1f} GB of {'speech timing' if cloud else 'local AI'} files to enable editing.")
        elif cloud and not self.models_ready:
            self.stage.setText('API key needed')
            self.status.setText('Set OPENAI_API_KEY in this computer’s environment, then restart. OpenAI mode uploads commentary audio and transcript context.')
        elif cloud:
            self.stage.setText('Ready to analyze')
            self.status.setText('OpenAI will receive commentary audio and transcript context. Review and restore cuts before exporting.')
        else:
            self.stage.setText('Ready');self.status.setText('Local processing. Open a Resolve project before starting.')
        return dict(readiness,ready=self.models_ready)
    def start_model_setup(self):
        if self.setup_worker and self.setup_worker.isRunning():return
        self.setup_button.setEnabled(False);self.create.setEnabled(False);self.choose.setEnabled(False)
        self.stage.setText('Setting up local AI');self.status.setText('Checking downloaded files…');self.dial.set_busy(True)
        self.backend.setEnabled(False)
        self.setup_worker=ModelSetupWorker(groups=['speech'] if self.backend.currentData()=='openai' else None)
        self.setup_worker.progress.connect(self.status.setText)
        self.setup_worker.ready.connect(self.model_setup_done)
        self.setup_worker.failed.connect(self.model_setup_failed)
        self.setup_worker.finished.connect(self.model_setup_finished);self.setup_worker.start()
    def model_setup_done(self,result):
        self.models_ready=result['ready'] and (self.backend.currentData()!='openai' or bool(os.environ.get('OPENAI_API_KEY','').strip()));self.setup_button.hide()
        self.stage.setText('Ready' if self.models_ready else 'API key needed')
        self.status.setText('Speech timing is installed. Set OPENAI_API_KEY and restart to use OpenAI.' if not self.models_ready else 'Models are installed. Choose a recording to begin.')
    def model_setup_failed(self,message):
        self.models_ready=False;self.stage.setText('Setup paused');self.status.setText(message)
        self.setup_button.setText('Retry setup');self.setup_button.show()
    def model_setup_finished(self):
        self.dial.set_busy(False);self.setup_button.setEnabled(True);self.choose.setEnabled(True);self.backend.setEnabled(True)
        self.create.setEnabled(self.source is not None and self.models_ready)
        if self.close_after_setup:self.close()
    def pick(self):
        f,_=QFileDialog.getOpenFileName(self,'Choose your full recording',str(self.source.parent if self.source else ROOT),'Video recordings (*.mp4 *.mov *.mkv *.m4v)')
        if f:self.load(Path(f))
    def load(self,path):
        try:
            duration,fps,names=probe(path)
            self.source=path;self.voice.clear();self.voice.addItems(names)
            mic=next((i for i,n in enumerate(names) if 'microphone' in n.lower() or 'mic'==n.lower()),None)
            if mic is not None:self.voice.setCurrentIndex(mic)
            ambiguous=mic is None and len(names)>1
            self.voice.setVisible(len(names)>1);self.voice_label.setVisible(len(names)>1)
            voice=names[self.voice.currentIndex()]
            self.dial.set_duration(duration)
            self.file.setText(f'{path.name}\n{voice}')
            self.stage.setText('Ready');self.choose.setText('Change');self.status.setText('Open a Resolve project before starting.')
            self.refresh_models()
            save_json(ROOT/'analysis/simple-preferences.json',dict(source=str(path),intensity=self.intensity.currentData(),backend=self.backend.currentData()))
        except Exception as e:self.source=None;self.status.setText('Could not open that recording: '+str(e));self.create.setEnabled(False)
    def begin(self):
        if self.worker and self.worker.isRunning():return
        if self.setup_worker and self.setup_worker.isRunning():return
        if self.source is None:return
        if not self.refresh_models()['ready']:return
        save_json(ROOT/'analysis/simple-preferences.json',dict(source=str(self.source),intensity=self.intensity.currentData(),backend=self.backend.currentData()))
        self.result=None;self.choose.setEnabled(False);self.voice.setEnabled(False);self.intensity.setEnabled(False);self.backend.setEnabled(False);self.create.setEnabled(False)
        self.dial.set_busy(True)
        self.create.setText('Creating your cut…');self.stage.setText('Creating your edit');self.status.setText('Preparing edit proposals…' if self.backend.currentData()=='openai' else 'Starting local analysis…')
        intensity=self.intensity.currentData()
        if self.backend.currentData()=='local' and intensity=='aggressive':intensity='tight'
        self.worker=Worker(self.source,self.voice.currentIndex(),intensity,self.backend.currentData());self.worker.progress.connect(self.status.setText)
        self.worker.ready.connect(self.done);self.worker.failed.connect(self.failed);self.worker.finished.connect(self.finished);self.worker.start()
    def done(self,result):
        if result.get('ai_plan') and not result.get('timeline_verified') and not result.get('manual_import_required'):
            from ai_review_ui import ReviewDialog
            self.result=result;self.stage.setText('Your proposals are ready')
            self.dial.set_busy(False);self.dial.set_duration(result['edited_duration'],'Proposed length')
            self.status.setText('Review proposed cuts, compare the audio, and restore or adjust them before exporting to Resolve.')
            self.review_dialog=ReviewDialog(result,self)
            self.review_dialog.exported.connect(self.done);self.review_dialog.show()
            return
        self.stage.setText('Your draft is ready')
        self.result=result;d=round(result['edited_duration']);o=round(result['duration'])
        self.dial.set_busy(False);self.dial.set_duration(d,'Draft length')
        review=result.get('script_review',{})
        flag_count=len(review.get('flags',[]))+len(result.get('evidence_flags',[]))
        self.status.setText(f'{o//60}:{o%60:02} → {d//60}:{d%60:02}  ·  {result["clips"]} clips\nReview in Resolve. {flag_count} checks flagged. Render and listening review still required.')
        if result.get('comparison',{}).get('identical_frame_ranges'):
            self.status.setText(self.status.text()+'\nFresh analysis produced the same cut decisions.')
        if result.get('manual_import_required'):
            self.stage.setText('Draft saved — import into Resolve')
            instruction=(f"In Resolve, choose File → Import → Timeline and select {Path(result['import_xml']).name}." if result.get('import_xml') else 'Follow OPEN IN RESOLVE.txt in the draft folder to import the edit.')
            self.status.setText(f'{o//60}:{o%60:02} → {d//60}:{d%60:02}  ·  {result["clips"]} clips\n{instruction} Open the draft folder below.')
        if result.get('folder'):
            self.details.setText('<a style="color:#c6ddfa" href="draft">Open draft files</a>')
    def open_export_folder(self,*args):
        if self.result and self.result.get('folder'):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.result['folder']))
    def failed(self,message):self.stage.setText('Needs your attention');self.status.setText(message)
    def finished(self):
        self.dial.set_busy(False)
        self.choose.setEnabled(True);self.voice.setEnabled(True);self.intensity.setEnabled(True);self.backend.setEnabled(True);self.create.setText('Create automatic cut');self.create.setEnabled(self.source is not None and self.models_ready)
        if '--auto-check' in sys.argv:
            save_json(ROOT/'analysis/automatic-ui-check.json',dict(success=bool(self.result),result=self.result,status=self.status.text(),button=self.create.text()))
            def finish_check():
                self.grab().save(str(ROOT/'analysis/automatic-ui-check.png'));QApplication.instance().quit()
            QTimer.singleShot(500,finish_check)
    def closeEvent(self,event):
        if self.review_dialog and self.review_dialog.busy():
            event.ignore();self.status.setText('Finish the active review render or Resolve import before closing.');return
        if self.setup_worker and self.setup_worker.isRunning():
            event.ignore();self.close_after_setup=True;self.setup_worker.requestInterruption()
            self.status.setText('Stopping setup safely… Downloaded files will be reused next time.');return
        if self.worker and self.worker.isRunning():
            event.ignore();self.status.setText('The cut is still being created. Keep this window open until it finishes.');return
        super().closeEvent(event)

def main():
    from PySide6.QtNetwork import QLocalServer,QLocalSocket
    from launch_status import report_startup
    app=QApplication(sys.argv);configure_app(app)
    checking='--launch-check' in sys.argv or '--auto-check' in sys.argv
    name='retention-cut-'+hashlib.sha1(str(ROOT).encode()).hexdigest()[:12]
    if checking:name+='-'+str(os.getpid())
    socket=QLocalSocket();socket.connectToServer(name)
    if socket.waitForConnected(400):
        socket.write(b'show');socket.flush()
        # A connected pipe alone is not a successful launch. The existing GUI
        # must process the request, restore its window, and acknowledge it.
        if not socket.bytesAvailable() and not socket.waitForReadyRead(3000):
            raise RuntimeError('The running editor did not respond. See analysis/application.log.')
        try:ack=json.loads(bytes(socket.readAll()).decode('utf-8'))
        except (ValueError,UnicodeError) as error:
            raise RuntimeError('The running editor returned an invalid startup response.') from error
        if not ack.get('visible') or ack.get('state')!='ready':
            raise RuntimeError('The running editor could not restore its window.')
        report_startup('existing',visible=True,existing_pid=ack.get('pid'))
        return 0
    server=QLocalServer()
    if not server.listen(name):
        raise RuntimeError('Could not create the editor startup service: '+server.errorString())
    window=AutomaticWindow();window.show()
    def reveal():
        client=server.nextPendingConnection()
        if not client:return
        window.showNormal();window.raise_();window.activateWindow()
        client.disconnected.connect(client.deleteLater)
        client.write(json.dumps(dict(state='ready',pid=os.getpid(),visible=window.isVisible())).encode('utf-8'))
        client.flush();client.disconnectFromServer()
    server.newConnection.connect(reveal)
    # Execute after the event loop has processed the initial window events.
    def startup_ready():
        snapshot=os.environ.get('RETENTION_STARTUP_SNAPSHOT')
        if snapshot:
            destination=Path(snapshot);destination.parent.mkdir(parents=True,exist_ok=True)
            if not window.grab().save(str(destination)):
                report_startup('error',error='Could not save the requested startup verification snapshot')
                return
        report_startup('ready',visible=window.isVisible(),title=window.windowTitle())
    QTimer.singleShot(300,startup_ready)
    if '--launch-check' in sys.argv:
        def check():
            save_json(ROOT/'analysis/automatic-launch-check.json',dict(visible=window.isVisible(),title=window.windowTitle(),button=window.create.text(),old_review_buttons=False,executable=sys.executable))
            window.grab().save(str(ROOT/'analysis/automatic-launch-check.png'));app.quit()
        QTimer.singleShot(1200,check)
    if '--auto-check' in sys.argv:QTimer.singleShot(500,window.create.click)
    return app.exec()

if __name__=='__main__':sys.exit(main())
