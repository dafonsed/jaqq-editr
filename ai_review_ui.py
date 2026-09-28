"""Review validated AI proposals before importing a new Resolve timeline."""
import copy
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QKeySequence, QShortcut, QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox, QSlider, QComboBox)

from app_paths import ROOT
from cut_review import VideoPreview
from review_core import save_json, stamp


def review_folder(result):
    return Path(result['plan_path']).parent if result.get('plan_path') else Path(result.get('analysis_folder', ROOT/'analysis'/'ai-review'))


class RenderWorker(QThread):
    progress = Signal(str)
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, result, ranges, parent=None):
        super().__init__(parent)
        self.result, self.ranges = copy.deepcopy(result), ranges

    def run(self):
        try:
            from review_render import render_preview, audit_render, transcribe_render
            folder = review_folder(self.result)
            path = folder/f'review-{time.time_ns()}.mp4'
            words=[word for segment in self.result.get('transcript',[]) for word in segment.get('words',[])]
            data = render_preview(self.result['source'], self.ranges, self.result['duration'],
                self.result['fps'], path, progress=self.progress.emit,
                cancel=self.isInterruptionRequested, annotations=words)
            data.update(source_key=self.result.get('source_key'),cut_signature=self.result.get('cut_signature'),
                        review_scope='full' if self.ranges==self.result['kept'] else 'context')
            try:
                observed=transcribe_render(path,self.result.get('microphone',0),data['output_frames']/data['fps'],
                    progress=self.progress.emit,cancel=self.isInterruptionRequested)
            except InterruptedError:
                raise
            except Exception:
                observed=dict(status='NOT VERIFIED',reason='Rendered speech transcription failed; timing/acoustic checks remain available.',words=[])
            data['render_transcription']=observed
            data['qa'] = audit_render(path, data['frame_map'], self.result['fps'],
                microphone=self.result.get('microphone', 0),
                long_pause=self.result.get('settings', {}).get('audit_gap', .9),
                expected_words=data['annotations'] if words else None,
                observed_words=observed['words'] if observed['status']=='TRANSCRIBED' else None,
                progress=self.progress.emit,cancel=self.isInterruptionRequested)
            self.ready.emit(data)
        except Exception as error:
            self.failed.emit(str(error))


class SendWorker(QThread):
    progress = Signal(str)
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = copy.deepcopy(result)

    def run(self):
        try:
            from automatic_cut import send
            self.ready.emit(send(self.result, self.progress.emit))
        except Exception as error:
            self.failed.emit(str(error))


class ReviewDialog(QDialog):
    exported = Signal(dict)

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Review proposed edits')
        self.setWindowModality(Qt.WindowModal)
        self.resize(1120, 900)
        self.original = copy.deepcopy(result)
        self.result = copy.deepcopy(result)
        self.overrides = copy.deepcopy(result.get('ai_plan', {}).get('overrides', {}))
        self.history = []
        self.render_worker = self.send_worker = None
        self.stop_at = self.pending_seek = None
        self.pending_play = False
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Source', 'Action', 'State', 'Confidence', 'Model', 'Reason', 'ID'])
        self.table.setColumnHidden(6, True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        for column,width in enumerate([165,210,80,85,115]):
            self.table.setColumnWidth(column,width)
        self.table.currentCellChanged.connect(self.select)
        layout.addWidget(self.table, 2)
        self.reason = QLabel('Select a proposal to inspect its source context.')
        self.reason.setTextFormat(Qt.PlainText)
        self.reason.setWordWrap(True)
        layout.addWidget(self.reason)
        controls = QHBoxLayout()
        self.start, self.end = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (self.start, self.end):
            widget.setDecimals(4)
            widget.setRange(0, result['duration'])
            widget.setSingleStep(1/result['fps'])
            widget.setSuffix(' s')
        controls.addWidget(QLabel('Removal boundary'))
        controls.addWidget(self.start)
        controls.addWidget(self.end)
        self.adjust_button = self.button('Validate adjustment', self.adjust)
        self.restore_button = self.button('Restore cut', lambda: self.override(False))
        self.apply_button = self.button('Reapply cut', lambda: self.override(True))
        self.undo_button = self.button('Undo', self.undo)
        for widget in (self.adjust_button, self.restore_button, self.apply_button, self.undo_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.video = VideoPreview()
        self.video.setMinimumSize(480, 240)
        layout.addWidget(self.video, 3)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.8)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoSink(self.video.videoSink())
        self.player.positionChanged.connect(self.position_changed)
        self.player.mediaStatusChanged.connect(self.media_status)
        self.player.tracksChanged.connect(self.tracks_changed)
        self.player.errorOccurred.connect(lambda *_: self.status.setText('Playback: '+self.player.errorString()))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.seek)
        layout.addWidget(self.slider)
        playback = QHBoxLayout()
        self.before_button = self.button('Play original context', self.before)
        self.after_button = self.button('Render / play edited context', self.after)
        self.pause_button = self.button('Play / Pause', self.toggle)
        self.full_button = self.button('Render full review', self.full_preview)
        for widget in (self.before_button, self.after_button, self.pause_button, self.full_button):
            playback.addWidget(widget)
        self.listen = QComboBox()
        self.listen.currentIndexChanged.connect(self.player.setActiveAudioTrack)
        playback.addWidget(QLabel('Listen'))
        playback.addWidget(self.listen)
        layout.addLayout(playback)
        self.status = QLabel('Check meaning, breaths and visual continuity. Speaker identity is not automatically guaranteed.')
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        footer.addWidget(QLabel('Invalid or uncertain proposals remain in the recording.'))
        footer.addStretch()
        self.export_button = self.button('Send reviewed draft to Resolve', self.export)
        self.open_export_button = self.button('Open saved export',self.open_export)
        self.open_export_button.hide()
        footer.addWidget(self.open_export_button)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)
        QShortcut(QKeySequence('Ctrl+Z'), self, activated=self.undo)
        self.refresh()

    def button(self, text, handler):
        button = QPushButton(text)
        button.clicked.connect(handler)
        return button

    def decisions(self):
        return self.result.get('ai_plan', {}).get('decisions', [])

    def current(self):
        index = self.table.currentRow()
        return self.decisions()[index] if 0 <= index < len(self.decisions()) else None

    def refresh(self):
        current = max(0, self.table.currentRow())
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.decisions()))
        for index, item in enumerate(self.decisions()):
            override = self.overrides.get(item['candidate_id'], {})
            state = 'restored' if override.get('enabled') is False else item.get('status', 'review')
            values = [f"{stamp(item['start'])} – {stamp(item['end'])}", item['action'], state,
                f"{item.get('confidence', 0):.0%}", item.get('model') or 'No model call',
                item.get('reason', ''), item['candidate_id']]
            for column, value in enumerate(values):
                cell=QTableWidgetItem(str(value));cell.setToolTip(str(value))
                self.table.setItem(index, column, cell)
            self.table.setRowHeight(index,48)
        self.table.blockSignals(False)
        length = sum(round(b*self.result['fps'])-round(a*self.result['fps']) for a, b in self.result['kept'])/self.result['fps']
        metrics = self.result.get('ai_plan', {}).get('metrics', {})
        billing=metrics.get('video_total',metrics)
        cost = billing.get('cost_usd_estimate',billing.get('total_cost_usd',billing.get('estimated_cost_usd')))
        cost_text = f'Estimated API cost ${cost:.4f}' if isinstance(cost, (float, int)) else 'API usage/cost details are saved with the plan'
        self.summary.setText(f"{Path(self.result['source']).name} · {stamp(self.result['duration'])} → {stamp(length)} · "
            f"{len(self.decisions())} proposals. {cost_text}.\nConfidence is model-reported; audition accepted cuts before export.")
        if self.decisions():
            self.table.selectRow(min(current, len(self.decisions())-1))
            self.select()
        else:
            for widget in (self.adjust_button, self.restore_button, self.apply_button, self.before_button, self.after_button):
                widget.setEnabled(False)
        self.undo_button.setEnabled(bool(self.history))

    def select(self, *_):
        item = self.current()
        if item is None:
            return
        override = self.overrides.get(item['candidate_id'], {})
        self.start.setValue(override.get('start', item['start']))
        self.end.setValue(override.get('end', item['end']))
        self.reason.setText(item.get('reason', '')+'\n'+str(item.get('validation_reason', item.get('required_context', ''))))
        # Review/rejected decisions cannot be activated by clicking through a UI.
        original = next((d for d in self.original['ai_plan'].get('original_decisions', self.original['ai_plan']['decisions'])
                         if d['candidate_id']==item['candidate_id']), item)
        accepted = original.get('status')=='accepted' and original['action'] not in ('KEEP', 'NEEDS_REVIEW')
        self.adjust_button.setEnabled(accepted)
        self.restore_button.setEnabled(accepted and override.get('enabled') is not False)
        self.apply_button.setEnabled(accepted and override.get('enabled') is False)

    def commit_overrides(self, overrides):
        from ai_pipeline import rebuild_plan
        try:
            revised = rebuild_plan(self.original, overrides)
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText('Adjustment rejected: '+str(error))
            return False
        self.history.append(copy.deepcopy(self.overrides))
        self.overrides, self.result = overrides, revised
        self.player.pause()
        self.refresh()
        self.save()
        self.status.setText('Validated and saved. Render the changed join again before exporting.')
        return True

    def override(self, enabled):
        item = self.current()
        if item is None:
            return
        changed = copy.deepcopy(self.overrides)
        changed[item['candidate_id']] = dict(changed.get(item['candidate_id'], {}), enabled=enabled)
        self.commit_overrides(changed)

    def adjust(self):
        item = self.current()
        if item is None:
            return
        fps = self.result['fps']
        changed = copy.deepcopy(self.overrides)
        changed[item['candidate_id']] = dict(enabled=True,
            start=round(self.start.value()*fps)/fps, end=round(self.end.value()*fps)/fps)
        self.commit_overrides(changed)

    def undo(self):
        if not self.history:
            return
        from ai_pipeline import rebuild_plan
        previous = self.history[-1]
        try:
            revised = rebuild_plan(self.original, previous)
        except ValueError as error:
            self.status.setText('Undo could not be validated: '+str(error))
            return
        self.history.pop()
        self.overrides, self.result = previous, revised
        self.refresh()
        self.save()
        self.status.setText('Last adjustment undone.')

    def save(self):
        folder = review_folder(self.result)
        save_json(folder/'reviewed-plan.json', self.result)
        save_json(folder/'review-overrides.json', self.overrides)

    def tracks_changed(self):
        count = len(self.player.audioTracks())
        self.listen.blockSignals(True)
        self.listen.clear()
        self.listen.addItems([f'Audio {i+1}' for i in range(count)])
        selected = min(self.result.get('microphone', 0), count-1)
        self.listen.setCurrentIndex(selected)
        self.listen.blockSignals(False)
        if selected >= 0:
            self.player.setActiveAudioTrack(selected)

    def media_status(self, state):
        if state in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia) and self.pending_play:
            self.pending_play = False
            self.player.setPosition(round((self.pending_seek or 0)*1000))
            self.pending_seek = None
            self.player.play()

    def play(self, path, start, stop):
        target = QUrl.fromLocalFile(str(Path(path).resolve()))
        self.stop_at = stop
        self.slider.setMaximum(round(stop*1000))
        if target != self.player.source():
            self.pending_seek, self.pending_play = start, True
            self.player.setSource(target)
        else:
            self.player.setPosition(round(start*1000))
            self.player.play()

    def before(self):
        item = self.current()
        if item:
            self.play(self.result['source'], max(0, item['start']-1.5), min(self.result['duration'], item['end']+1.5))

    def after(self):
        item = self.current()
        if item:
            from review_render import preview_ranges
            self.render(preview_ranges(self.result, item['start'], item['end']))

    def full_preview(self):
        self.render(self.result['kept'])

    def render(self, ranges):
        if self.busy():
            return
        if not ranges:
            self.status.setText('No retained picture exists in this preview window. Restore the cut to compare it.')
            return
        self.player.pause()
        self.render_worker = RenderWorker(self.result, ranges, self)
        self.set_busy(True)
        self.render_worker.progress.connect(self.status.setText)
        self.render_worker.ready.connect(self.rendered)
        self.render_worker.failed.connect(lambda message: self.status.setText('Review render failed: '+message))
        self.render_worker.finished.connect(lambda: self.set_busy(False))
        self.render_worker.start()

    def rendered(self, data):
        self.result['review_render'] = data
        self.save()
        qa = data['qa']
        self.status.setText(f"Review rendered. Timing: {qa['geometry']}; {len(qa['issues'])} acoustic/timing flags; "
            f"word comparison: {qa['word_check']['status']}. Audio uses the original hard joins. "
            'Listen for clipped words and breaths, and inspect visual joins. Listening remains unverified.')
        self.play(data['output'], 0, data['output_frames']/data['fps'])

    def export(self):
        if self.busy():
            return
        # Re-run the server-side edit validator immediately before export.
        from ai_pipeline import rebuild_plan
        try:
            self.result = rebuild_plan(self.result, self.overrides)
        except ValueError as error:
            self.status.setText('Export blocked: '+str(error))
            return
        self.player.pause()
        self.save()
        self.send_worker = SendWorker(self.result, self)
        self.set_busy(True)
        self.send_worker.progress.connect(self.status.setText)
        self.send_worker.ready.connect(self.sent)
        self.send_worker.failed.connect(lambda message: self.status.setText('Resolve export: '+message))
        self.send_worker.finished.connect(lambda: self.set_busy(False))
        self.send_worker.start()

    def sent(self, result):
        self.result = result
        self.open_export_button.setVisible(bool(result.get('folder')))
        if result.get('manual_import_required'):
            self.status.setText('Draft files are saved. Automatic Resolve import was unavailable. '
                'Open the saved export and follow OPEN IN RESOLVE.txt. The Resolve timeline and final render remain unverified.')
            return
        if not result.get('timeline_verified'):
            self.status.setText('Draft files are saved, but Resolve did not confirm a verified timeline.')
            return
        self.status.setText('New draft imported into Resolve. Final Resolve render and listening review remain required.')
        self.exported.emit(result)

    def open_export(self):
        if self.result.get('folder'):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.result['folder']).resolve())))

    def busy(self):
        return any(worker is not None and worker.isRunning() for worker in (self.render_worker, self.send_worker))

    def set_busy(self, value):
        for widget in (self.table, self.start, self.end, self.adjust_button, self.restore_button,
                       self.apply_button, self.undo_button, self.export_button, self.after_button, self.full_button):
            widget.setEnabled(not value)
        if not value:
            self.select()
            self.undo_button.setEnabled(bool(self.history))

    def position_changed(self, position):
        if self.stop_at is not None and position >= self.stop_at*1000:
            self.player.pause()
        if not self.slider.isSliderDown():
            self.slider.setValue(position)

    def seek(self, value):
        self.player.setPosition(value)

    def toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def closeEvent(self, event):
        if self.busy():
            event.ignore()
            if self.render_worker and self.render_worker.isRunning():
                self.render_worker.requestInterruption()
                self.status.setText('Stopping the review render safely. Close again after it stops.')
            else:
                self.status.setText('Finishing the Resolve import. Close after it completes.')
            return
        self.player.stop()
        super().closeEvent(event)
