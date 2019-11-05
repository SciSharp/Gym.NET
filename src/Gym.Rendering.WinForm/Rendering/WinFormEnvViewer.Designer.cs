namespace Gym.Rendering.WinForm
{
    partial class WinFormEnvViewer
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing); 
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.PictureFrame = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.PictureFrame)).BeginInit();
            this.SuspendLayout();
            // 
            // PictureFrame
            // 
            this.PictureFrame.Dock = System.Windows.Forms.DockStyle.Fill;
            this.PictureFrame.Location = new System.Drawing.Point(0, 0);
            this.PictureFrame.Name = "PictureFrame";
            this.PictureFrame.Size = new System.Drawing.Size(800, 450);
            this.PictureFrame.TabIndex = 0;
            this.PictureFrame.TabStop = false;
            // 
            // Viewer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.PictureFrame);
            this.Name = "Viewer";
            this.Text = "Viewer";
            ((System.ComponentModel.ISupportInitialize)(this.PictureFrame)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        public System.Windows.Forms.PictureBox PictureFrame;
    }
}