.PHONY: help test-psd export-roi clean

# Colors
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
RED     := \033[31m
MAGENTA := \033[35m
BOLD    := \033[1m
DIM     := \033[2m
RESET   := \033[0m

# Default target - show help
help:
	@echo ""
	@echo "$(BOLD)$(MAGENTA)  ╔═══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BOLD)$(MAGENTA)  ║$(RESET)$(BOLD)         🧠  Mouse EEG Source Localization  🧠            $(MAGENTA)║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)  ╚═══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "  $(BOLD)$(CYAN)COMMANDS$(RESET)"
	@echo "  $(DIM)────────────────────────────────────────────────────────────$(RESET)"
	@echo ""
	@echo "  $(GREEN)make test-psd EEG=<file>$(RESET)    Run PSD sanity check"
	@echo "  $(DIM)                            Validates 1/f spectral slope$(RESET)"
	@echo "  $(DIM)                            preservation in source localization$(RESET)"
	@echo ""
	@echo "  $(GREEN)make export-roi OUT=<file>$(RESET)  Export ROI time series to .set"
	@echo "  $(DIM)                            ROIs become channels in EEGLAB format$(RESET)"
	@echo ""
	@echo "  $(YELLOW)make clean$(RESET)                 Remove generated test outputs"
	@echo ""
	@echo "  $(DIM)────────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(BOLD)$(CYAN)EXAMPLES$(RESET)"
	@echo ""
	@printf "  $(DIM)%s$(RESET) make test-psd EEG=/path/to/recording.set\n" "$$"
	@printf "  $(DIM)%s$(RESET) make export-roi OUT=source_rois.set\n" "$$"
	@echo ""

# Run PSD sanity check test
test-psd:
ifndef EEG
	@echo ""
	@echo "  $(RED)$(BOLD)ERROR$(RESET) $(RED)EEG file not specified$(RESET)"
	@echo ""
	@echo "  $(DIM)Usage:$(RESET) make test-psd EEG=/path/to/file.set"
	@echo ""
	@exit 1
endif
	@echo ""
	@echo "  $(CYAN)$(BOLD)▶ Running PSD Sanity Check$(RESET)"
	@echo "  $(DIM)────────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(DIM)Input:$(RESET) $(EEG)"
	@echo ""
	@. .venv/bin/activate && python tests/test_psd_sanity.py $(EEG) && \
		echo "" && \
		echo "  $(GREEN)$(BOLD)✓ Complete$(RESET)" && \
		echo "" || \
		(echo "" && echo "  $(RED)$(BOLD)✗ Failed$(RESET)" && echo "" && exit 1)

# Export ROI time series to EEGLAB .set format
export-roi:
ifndef OUT
	@echo ""
	@echo "  $(RED)$(BOLD)ERROR$(RESET) $(RED)Output file not specified$(RESET)"
	@echo ""
	@echo "  $(DIM)Usage:$(RESET) make export-roi OUT=/path/to/output.set"
	@echo ""
	@exit 1
endif
	@echo ""
	@echo "  $(CYAN)$(BOLD)▶ Exporting ROI Time Series$(RESET)"
	@echo "  $(DIM)────────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(DIM)Output:$(RESET) $(OUT)"
	@echo ""
	@. .venv/bin/activate && python -c "\
from src.source_localization.utils.export_set import export_roi_to_set; \
from src.source_localization.utils.io_utils import load_pickle; \
roi_data = load_pickle('test_results/data/step6_roi_timeseries.pkl'); \
export_roi_to_set(roi_data, sfreq=500.0, output_path='$(OUT)')" && \
		echo "" && \
		echo "  $(GREEN)$(BOLD)✓ Complete$(RESET)" && \
		echo "" || \
		(echo "" && echo "  $(RED)$(BOLD)✗ Failed$(RESET)" && echo "" && exit 1)

# Clean generated outputs
clean:
	@echo ""
	@echo "  $(YELLOW)$(BOLD)▶ Cleaning test outputs...$(RESET)"
	@rm -f test_results/psd_sanity_check.png
	@rm -f test_results/psd_trial_averaged_*.png
	@echo "  $(GREEN)$(BOLD)✓ Done$(RESET)"
	@echo ""
