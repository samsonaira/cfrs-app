"""
Management command: train the CFRS recommendation models.

    python manage.py train_recommender

Loads transaction data, preprocesses it, trains user-based CF, item-based
CF and a popularity baseline, evaluates them all on a held-out test set,
and saves model artifacts + real evaluation results to ml/models/.
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Train and evaluate the recommendation models, saving artifacts."

    def handle(self, *args, **options):
        from ml.training.train import train_all

        self.stdout.write("Training CFRS recommendation models...")
        results = train_all(verbose=False)

        ds = results["dataset"]
        self.stdout.write(self.style.SUCCESS("\nDataset:"))
        self.stdout.write(
            f"  buyers={ds['n_buyers']} farmers={ds['n_farmers']} "
            f"interactions={ds['n_interactions']} sparsity={ds['sparsity']}"
        )
        self.stdout.write(self.style.SUCCESS("\nModel evaluation (held-out test set):"))
        self.stdout.write(
            f"  {'Model':<24}{'RMSE':>8}{'MAE':>8}{'P@5':>8}{'R@5':>8}{'F1@5':>8}"
        )
        for m in results["models"]:
            self.stdout.write(
                f"  {m['model']:<24}{m['rmse']:>8}{m['mae']:>8}"
                f"{m['precision_at_5']:>8}{m['recall_at_5']:>8}{m['f1_at_5']:>8}"
            )
        self.stdout.write(self.style.SUCCESS("\nArtifacts saved to ml/models/"))
